# Empirical Evaluation for Neural Causal Models

This repository is built on the codebase from the paper **“Neural Causal Models for Counterfactual Identification and Estimation”** by **Kevin Xia, Yushu Pan, and Elias Bareinboim**.

It extends that codebase to run **empirical evaluation experiments comparing EM4CI and Neural Causal Models (NCMs)**. The results are reported in the paper **“An Empirical Evaluation of Model Completion for Causal Inference”** by **Jiapeng Zhao, Elias Bareinboim, and Rina Dechter**.

Below are instructions to run this code and reproduce the experimental results.

---

## 1. Setup

Install dependencies:

```
python -m pip install -r requirements.txt
```

---

## 2. Running the code

Both **identification** and **estimation** procedures can be run through `src/main.py`.

### 2.1 Identification (GAN-NCM)

From the repository root:

```
python -m src.main <NAME> gan \
  --lr 2e-5 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm \
  --gan-mode wgangp --d-iters 1 --id-query <QUERY> -r 4 \
  --max-lambda 1e-4 --min-lambda 1e-5 --max-query-iters 1000 \
  --single-disc --gen-sigmoid --mc-sample-size 256 \
  -G expl_set -t 20 -n 10000 -d <DIM> --gpu 0
```

- `<NAME>`: output experiment folder name
- `<QUERY>`: one of `ate`, `ett`, `nde`, `ctfde`
- `<DIM>`: dimensionality of `Z` (e.g., `1` or `16`)

### 2.2 Identification (MLE-NCM)

```
python -m src.main <NAME> mle \
  --full-batch --h-size 64 --id-query <QUERY> -r 4 \
  --max-query-iters 1000 --mc-sample-size 10000 \
  -G expl_set -t 20 -n 10000 -d <DIM> --gpu 0
```

---

## 3. Reproducing the empirical evaluation scripts (8 small graphs + 6 large models)

We use **categorical / one-hot mode** through:

- `--domain-sizes K` (domain size `K` for each observed variable)
- `--gumbel-tau` (Gumbel-Softmax temperature used by the categorical models)

> Note: Commands below are representative examples used in our experiments. For a complete record, see `scrip_note.sh`.

### 3.1 GAN-NCM (8exp, 10 trials, categorical/one-hot)

`n=100`:

```
python -m src.main est_8exp_cpt_onehot1 gan \
  --gen CPT --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm \
  --gan-mode wgangp --d-iters 1 --query-track avg_error \
  --single-disc --mc-sample-size 5000 \
  -G 8exp -t 10 -n 100 -d 1 \
  --domain-sizes 2 --gumbel-tau 1.0 --gpu 0
```

`n=1000`:

```
python -m src.main est_8exp_cpt_onehot1 gan \
  --gen CPT --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm \
  --gan-mode wgangp --d-iters 1 --query-track avg_error \
  --single-disc --mc-sample-size 5000 \
  -G 8exp -t 10 -n 1000 -d 1 \
  --domain-sizes 2 --gumbel-tau 1.0 --gpu 0
```

### 3.2 GAN-NCM (6large, 5 trials, categorical/one-hot)

`n=1000`:

```
python -m src.main est_6large_cpt_onehot1 gan \
  --gen CPT --lr 0.00002 --data-bs 1000 --ncm-bs 1000 --h-size 64 --u-size 2 --layer-norm \
  --gan-mode wgangp --d-iters 1 --query-track avg_error \
  --single-disc --gen-sigmoid --mc-sample-size 10000 \
  -G 6large -t 5 -n 1000 -d 1 \
  --domain-sizes 4 --gumbel-tau 1.0 --gpu 0
```

`n=10000`:

```
python -m src.main est_6large_cpt_onehot1 gan \
  --gen CPT --lr 0.00002 --data-bs 4096 --ncm-bs 4096 --h-size 64 --u-size 2 --layer-norm \
  --gan-mode wgangp --d-iters 1 --query-track avg_error \
  --single-disc --gen-sigmoid --mc-sample-size 10000 \
  -G 6large -t 5 -n 10000 -d 1 \
  --domain-sizes 4 --gumbel-tau 1.0 --gpu 0
```

### 3.3 MLE-NCM (8exp, 10 trials, categorical/one-hot)

`n=100`:

```
python -m src.main mle_8exp_cpt_onehot1 mle \
  --full-batch --gen CPT --h-size 64 --query-track avg_error \
  --max-query-iters 1000 --mc-sample-size 10000 \
  -G 8exp -t 10 -n 100 -d 1 \
  --gpu 0 --domain-sizes 2 --gumbel-tau 1.0
```

`n=1000`:

```
python -m src.main mle_8exp_cpt_onehot1 mle \
  --full-batch --gen CPT --h-size 64 --query-track avg_error \
  --max-query-iters 1000 --mc-sample-size 10000 \
  -G 8exp -t 10 -n 1000 -d 1 \
  --gpu 0 --domain-sizes 2 --gumbel-tau 1.0
```

### 3.4 MLE-NCM (6large, 5 trials, categorical/one-hot)

`n=1000`:

```
python -m src.main mle_6large_cpt_onehot1 mle \
  --data-bs 128 --ncm-bs 128 --gen CPT --h-size 64 --query-track avg_error \
  --max-query-iters 1000 --mc-sample-size 1000 \
  -G 6large -t 5 -n 1000 -d 1 \
  --gpu 0 --domain-sizes 4 --gumbel-tau 1.0
```

`n=10000`:

```
python -m src.main mle_6large_cpt_onthot1 mle \
  --data-bs 256 --ncm-bs 256 --gen CPT --h-size 64 --query-track avg_error \
  --max-query-iters 1000 --mc-sample-size 1000 \
  -G 6large -t 5 -n 10000 -d 1 \
  --gpu 0 --domain-sizes 4 --gumbel-tau 1.0
```

---

## 4. Notes

- Experiment outputs are written under `out/<NAME>/...`. Failed runs are moved under `err/<NAME>/...`.
- If you rerun the **same experiment name** with different flags, the code may load existing checkpoints and evaluate (instead of retraining). Use a new `<NAME>` to force a fresh run.

---

## License

MIT License. Please cite the original NCM paper if you use this code.
