import os
import glob
import shutil
import hashlib
import json
import pandas as pd  # Import pandas for reading CSV files

import numpy as np
import torch as T
import pytorch_lightning as pl

from src.metric import evaluation
from src.ds.causal_graph import CausalGraph
from src.scm.ctm import CTM
from src.scm.cpt import CPT
from src.scm.cpt import parse_smile_cpt
from src.scm.scm import expand_do
from src.scm.representation import index_to_onehot, maybe_onehot_to_index
from .base_runner import BaseRunner


class NCMRunner(BaseRunner):
    def __init__(self, pipeline, dat_model, ncm_model):
        super().__init__(pipeline, dat_model, ncm_model)

    def create_trainer(self, directory, gpu=None):
        # Use EMA-smoothed training loss for stable checkpointing / early-stopping.
        monitor_key = 'train_loss_ema'
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f'{directory}/checkpoints/', monitor=monitor_key)
        trainer_kwargs = dict(
            callbacks=[
                checkpoint,
                pl.callbacks.EarlyStopping(
                    monitor=monitor_key,
                    patience=self.pipeline.patience,
                    min_delta=self.pipeline.min_delta,
                    #check_on_train_epoch_end=True
                )
            ],
            max_epochs=self.pipeline.max_epochs,
            accumulate_grad_batches=1,
            logger=pl.loggers.TensorBoardLogger(f'{directory}/logs/'),
            log_every_n_steps=1,
        )
        if gpu is not None:
            trainer_kwargs.update(dict(accelerator="gpu", devices=[gpu] if isinstance(gpu, int) else gpu))
        return pl.Trainer(**trainer_kwargs), checkpoint

    def run(self, exp_name, cg_file, n, dim, trial_index, hyperparams=None, gpu=None,
            lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False):
        key = self.get_key(cg_file, n, dim, trial_index)
        d = 'out/%s/%s' % (exp_name, key)  # name of the output directory

        if hyperparams is None:
            hyperparams = dict()

        with self.lock(f'{d}/lock', lockinfo) as acquired_lock:
            if not acquired_lock:
                print('[locked]', d)
                return

            try:
                # categorical domain size (all observed variables share the same K)
                domain_k = hyperparams.get('domain-sizes', None)
                if domain_k is not None:
                    domain_k = int(domain_k)
                    if domain_k <= 1:
                        raise ValueError(f"--domain-sizes must be >= 2, got {domain_k}")

                # return if best.th is generated (i.e. training is already complete)
                if os.path.isfile(f'{d}/best.th'):
                    print('[done]', d)
                    cg = CausalGraph.read(cg_file)

                    # If a hyperparams.json exists from the original run, prefer it for rebuilding the model.
                    # This prevents shape mismatches when rerunning with different flags (e.g., adding --domain-sizes).
                    hp_path = f'{d}/hyperparams.json'
                    if os.path.isfile(hp_path):
                        try:
                            with open(hp_path, 'r') as f:
                                saved_hp = json.load(f)
                            # saved hyperparams are stringified; merge but keep do-var-list/eval-query from current run
                            for k, v in saved_hp.items():
                                if k in {'do-var-list', 'eval-query'}:
                                    continue
                                hyperparams[k] = v
                            # normalize domain-sizes if present
                            if hyperparams.get('domain-sizes', None) in ('None', '', 'null'):
                                hyperparams['domain-sizes'] = None
                        except Exception:
                            pass

                    # normalize domain_k after possible hyperparam override
                    domain_k = hyperparams.get('domain-sizes', None)
                    if domain_k is not None:
                        domain_k = int(domain_k)

                    # Prefer to load the exact dat_sets used during training.
                    dat_sets = None
                    dat_th_path = f'{d}/dat.th'
                    if os.path.isfile(dat_th_path):
                        dat_sets = T.load(dat_th_path, map_location='cpu')

                    # If dat.th is missing, fall back to loading from data file (if present).
                    if dat_sets is None:
                        graph_name = os.path.basename(cg_file).split(".")[0]
                        data_file_path = f'/home/NCMCounterfactuals_test/dat/data/{n}/{graph_name}_TD{domain_k}_10.csv'
                        if os.path.isfile(data_file_path):
                            print(f"Loading data from {data_file_path}")
                            df = pd.read_csv(data_file_path, delimiter='\t')
                            u_cols = [c for c in df.columns if str(c).startswith('U')]
                            if len(u_cols) > 0:
                                df = df.drop(columns=u_cols)
                            dat_sets = [{
                                col: T.tensor(
                                    df[col].map({'a': 0, 'b': 1, 'c': 2, 'd': 3,
                                                 'State0': 0, 'State1': 1, 'State2': 2, 'State3': 3}).values
                                ).unsqueeze(1)
                                for col in df.columns
                            }]
                        else:
                            # Last resort: reconstruct generator + sample
                            if self.dat_model is CTM:
                                dat_m = self.dat_model(
                                    cg,
                                    v_size={k: dim for k in cg},
                                    regions=hyperparams.get('regions', 20),
                                    c2_scale=hyperparams.get('c2-scale', 1.0),
                                    batch_size=hyperparams.get('gen-bs', 10000),
                                    seed=0
                                )
                            elif self.dat_model is CPT:
                                graph_name = os.path.basename(cg_file).split(".")[0]
                                true_model_path = f'/home/NCMCounterfactuals_test/dat/true_model/{n}/{graph_name}_TD{domain_k}_10.xdsl'
                                print(f"Loading CPT from {true_model_path}")
                                variables, cpt_tables, parents, state_sizes = parse_smile_cpt(true_model_path)
                                dat_m = self.dat_model(variables, cpt_tables, parents, state_sizes, seed=0)
                            else:
                                dat_m = self.dat_model(cg, dim=dim, seed=0)

                            dat_sets = []
                            for dat_do_set in hyperparams["do-var-list"]:
                                expand_do_set = {k: expand_do(v, n=n) for (k, v) in dat_do_set.items()}
                                dat_sets.append(dat_m(n=n, do=expand_do_set))

                    # If categorical mode is enabled, ensure internal representation is one-hot.
                    # IMPORTANT: avoid double-conversion. Only convert if tensors are index-shaped.
                    if domain_k is not None:
                        converted = []
                        for ds in dat_sets:
                            ds2 = {}
                            for (k, v) in ds.items():
                                v_idx = maybe_onehot_to_index(v)
                                if T.is_tensor(v_idx) and v_idx.dim() == 2 and v_idx.shape[1] == 1:
                                    ds2[k] = index_to_onehot(v_idx, domain_k)
                                else:
                                    ds2[k] = v
                            converted.append(ds2)
                        dat_sets = converted

                    # Build dat_m only for metrics/pipeline wiring; the pipeline uses dat_sets for dat distribution.
                    if self.dat_model is CTM:
                        dat_m = self.dat_model(
                            cg,
                            v_size={k: (domain_k if domain_k is not None else dim) for k in cg},
                            regions=hyperparams.get('regions', 20),
                            c2_scale=hyperparams.get('c2-scale', 1.0),
                            batch_size=hyperparams.get('gen-bs', 10000),
                            seed=0
                        )
                    elif self.dat_model is CPT:
                        graph_name = os.path.basename(cg_file).split(".")[0]
                        true_model_path = f'/home/NCMCounterfactuals_test/dat/true_model/{n}/{graph_name}_TD{domain_k}_10.xdsl'
                        variables, cpt_tables, parents, state_sizes = parse_smile_cpt(true_model_path)
                        dat_m = self.dat_model(variables, cpt_tables, parents, state_sizes, seed=0)
                    else:
                        dat_m = self.dat_model(cg, dim=dim, seed=0)

                    m = self.pipeline(dat_m, hyperparams["do-var-list"], dat_sets, cg, dim, hyperparams=hyperparams,
                                      ncm_model=self.ncm_model)
                    ckpt = T.load(f'{d}/best.th')
                    try:
                        # For lightning checkpoints or plain state_dict.
                        state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
                        m.load_state_dict(state_dict)
                    except Exception:
                        # Last resort: try non-strict load to at least compute metrics if possible.
                        m.load_state_dict(state_dict, strict=False)

                    results_2 = evaluation.compute_average_errors(
                        m.generator, m.ncm, n=100000, dat_dos=hyperparams["do-var-list"]
                    )
                    with open(f'{d}/results_2.json', 'w') as file:
                        json.dump(results_2, file)
                    return d  # <--- RETURN IMMEDIATELY, DO NOT RETRAIN

                # since training is not complete, delete all directory files except for the lock
                print('[running]', d)
                for file in glob.glob(f'{d}/*'):
                    if os.path.basename(file) != 'lock':
                        if os.path.isdir(file):
                            shutil.rmtree(file)
                        else:
                            try:
                                os.remove(file)
                            except FileNotFoundError:
                                pass

                # set random seed to a hash of the parameter settings for reproducibility
                seed = int(hashlib.sha512(key.encode()).hexdigest(), 16) & 0xffffffff
                T.manual_seed(seed)
                np.random.seed(seed)
                print('Key:', key)
                print('Seed:', seed)

                # generate data-generating model, data, and model
                print('Generating data')
                cg = CausalGraph.read(cg_file)
                if self.dat_model is CTM:
                    if hyperparams['query-track'] != "avg_error":
                        v_sizes = {k: 1 if k in {'X', 'Y', 'M', 'W'} else dim for k in cg}
                        dat_m = self.dat_model(cg, v_size=v_sizes, regions=hyperparams.get('regions', 20),
                                            c2_scale=hyperparams.get('c2-scale', 1.0),
                                            batch_size=hyperparams.get('gen-bs', 10000),
                                            seed=seed)
                    else:
                        dat_m = self.dat_model(cg, v_size={k: dim for k in cg}, regions=hyperparams.get('regions', 20),
                                            c2_scale=hyperparams.get('c2-scale', 1.0),
                                            batch_size=hyperparams.get('gen-bs', 10000),
                                            seed=seed)
                elif self.dat_model is CPT:
                    graph_name = os.path.basename(cg_file).split(".")[0]  # Extract graph name (e.g., 'ex1')
                    data_file_path = f'/home/NCMCounterfactuals_test/dat/true_model/{n}/{graph_name}_TD{domain_k}_10.xdsl'
                    print(f"Loading CPT from {data_file_path}")
                    variables, cpt_tables, parents, state_sizes = parse_smile_cpt(data_file_path)
                    dat_m = self.dat_model(variables, cpt_tables, parents, state_sizes, seed=seed)
                else:
                    dat_m = self.dat_model(cg, dim=dim, seed=seed)

                # Check if data file exists
                graph_name = os.path.basename(cg_file).split(".")[0]  # Extract graph name (e.g., 'ex1')
                data_file_path = f'/home/NCMCounterfactuals_test/dat/data/{n}/{graph_name}_TD{domain_k}_10.csv'
                #print(f"Checking for data file at: {data_file_path}")
                #print(os.path.isfile(data_file_path))
                if os.path.isfile(data_file_path):
                    print(f"Loading data from {data_file_path}")
                    # Load the data file
                    df = pd.read_csv(data_file_path, delimiter='\t')

                    # Filter out exogenous variables if present (U*) to match NCM outputs
                    u_cols = [c for c in df.columns if str(c).startswith('U')]
                    if len(u_cols) > 0:
                        df = df.drop(columns=u_cols)

                    # Adjust the data format to match the required structure
                    dat_sets = [{
                        col: T.tensor(df[col].map({'a': 0, 'b': 1, 'c':2, 'd':3, 'State0':0, 'State1':1, 'State2':2, 'State3':3}).values).unsqueeze(1)
                        for col in df.columns
                    }]

                    # If categorical mode is enabled, convert external index data to internal one-hot
                    if domain_k is not None:
                        def _maybe_to_onehot(val: T.Tensor) -> T.Tensor:
                            if not T.is_tensor(val):
                                val = T.as_tensor(val)
                            # already one-hot/soft
                            if val.dim() == 2 and val.shape[1] == domain_k:
                                return val.float()
                            # index
                            if (val.dim() == 2 and val.shape[1] == 1) or val.dim() == 1 or val.dim() == 0:
                                return index_to_onehot(val, domain_k)
                            return val

                        dat_sets = [
                            {k: _maybe_to_onehot(v) for (k, v) in ds.items()}
                            for ds in dat_sets
                        ]
                    #print(dat_sets)
                else:
                    print("Generating data as no pre-existing file was found.")
                    dat_sets = []
                    for dat_do_set in hyperparams["do-var-list"]:
                        expand_do_set = {k: expand_do(v, n=n) for (k, v) in dat_do_set.items()}
                        dat_sets.append(dat_m(n=n, do=expand_do_set))  # Generate data

                    # If generator produced index data and categorical mode is enabled, convert to one-hot
                    if domain_k is not None:
                        def _maybe_to_onehot(val: T.Tensor) -> T.Tensor:
                            if not T.is_tensor(val):
                                val = T.as_tensor(val)
                            if val.dim() == 2 and val.shape[1] == domain_k:
                                return val.float()
                            if (val.dim() == 2 and val.shape[1] == 1) or val.dim() == 1 or val.dim() == 0:
                                return index_to_onehot(val, domain_k)
                            return val

                        dat_sets = [
                            {k: _maybe_to_onehot(v) for (k, v) in ds.items()}
                            for ds in dat_sets
                        ]
                    #print(dat_sets)

                # Ensure dat_sets format is consistent
                print("Data loaded successfully. Format adjusted to match required structure.")

                m = self.pipeline(dat_m, hyperparams["do-var-list"], dat_sets, cg, dim, hyperparams=hyperparams,
                                  ncm_model=self.ncm_model)

                # print info
                print("Calculating metrics")
                stored_metrics = dict()
                start_metrics = None
                for i, dat_do_set in enumerate(hyperparams["do-var-list"]):
                    name = evaluation.serialize_do(dat_do_set)
                    stored_metrics["true_{}".format(name)] = evaluation.probability_table(
                        dat_m, n=1000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()})
                    stored_metrics["dat_{}".format(name)] = evaluation.probability_table(
                        dat_m, n=1000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()},
                        dat=dat_sets[i])
                    if hyperparams['query-track'] == "avg_error":
                        if hyperparams["dim"] == 1:
                            results = evaluation.compute_average_errors(m.generator, m.ncm, n=10000)
                        else:
                            results = evaluation.compute_average_errors_n_dims(m.generator, m.ncm, n=10000, dims=hyperparams["dim"])
                    #else:   
                        #start_metrics = evaluation.all_metrics(m.generator, m.ncm, hyperparams["do-var-list"], dat_sets,
                                                       #n=1000000, stored=stored_metrics,
                                                       #query_track=hyperparams['eval-query'])

                # Only store true query results if the query is not "avg_error"
                if hyperparams['query-track'] != "avg_error" and hyperparams['query-track'] is not None:
                    true_q = 'true_{}'.format(evaluation.serialize_query(hyperparams['eval-query']))
                    stored_metrics[true_q] = start_metrics[true_q]

                m.update_metrics(stored_metrics)

                # train model
                if gpu is None:
                    gpu = int(T.cuda.is_available())
                trainer, checkpoint = self.create_trainer(d, gpu)
                trainer.fit(m)
                ckpt = T.load(checkpoint.best_model_path)
                m.load_state_dict(ckpt['state_dict'])
                results = {}
                
                if hyperparams['query-track'] == "avg_error":
                    if hyperparams["dim"] == 1:
                        results = evaluation.compute_average_errors(m.generator, m.ncm, n=10000)
                    else:
                        results = evaluation.compute_average_errors_n_dims(m.generator, m.ncm, n=10000, dims=hyperparams["dim"])
                else:
                    results = evaluation.all_metrics(m.generator, m.ncm, hyperparams["do-var-list"], dat_sets,
                                                 n=1000000, query_track=hyperparams['eval-query'])
                print(results)

                # save results
                with open(f'{d}/results.json', 'w') as file:
                    json.dump(results, file)
                with open(f'{d}/hyperparams.json', 'w') as file:
                    new_hp = {k: str(v) for (k, v) in hyperparams.items()}
                    json.dump(new_hp, file)
                T.save(dat_sets, f'{d}/dat.th')
                T.save(m.state_dict(), f'{d}/best.th')

                # Return the output directory
                return d
            except Exception:
                # move out/*/* to err/*/*/#
                e = d.replace("out/", "err/").rsplit('-', 1)[0]
                e_index = len(glob.glob(e + '/*'))
                e += '/%s' % e_index
                os.makedirs(e.rsplit('/', 1)[0], exist_ok=True)
                shutil.move(d, e)
                print(f'moved {d} to {e}')
                #print(f'Exception occurred in {d}, but keeping all files unchanged.')
                raise
