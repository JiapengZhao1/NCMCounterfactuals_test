import itertools
import numpy as np
import torch as T
import torch.nn as nn

from src.scm.distribution.continuous_distribution import UniformDistribution
from src.scm.nn.gumbel_mlp import GumbelMLP
from src.scm.scm import SCM, expand_do
from src.ds.counterfactual import CTF
from src.scm.representation import index_to_onehot, onehot_to_index


class MLE_NCM(SCM):
    def __init__(self, cg, v_size={}, default_v_size=1, u_size={},
                 default_u_size=1, f={}, hyperparams=None, default_module=GumbelMLP):

        if hyperparams is None:
            hyperparams = dict()

        self.cg = cg

        # categorical domain size (all observed variables share the same K)
        self.domain_k = hyperparams.get('domain-sizes', None)
        if self.domain_k is not None:
            self.domain_k = int(self.domain_k)

        self.u_size = {k: u_size.get(k, default_u_size) for k in self.cg.c2}
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.cg}
        super().__init__(
            v=list(cg),
            f=nn.ModuleDict({
                k: f[k] if k in f else default_module(
                    {k: self.v_size[k] for k in self.cg.pa[k]},
                    {k: self.u_size[k] for k in self.cg.v2c2[k]},
                    self.v_size[k],
                    h_layers=hyperparams.get('h-layers', 2),
                    h_size=hyperparams.get('h-size', 128)
                )
                for k in cg}),
            pu=UniformDistribution(self.cg.c2, self.u_size))

    def convert_evaluation(self, samples):
        # categorical mode: internal representation is one-hot/soft; external is index [n,1]
        if self.domain_k is not None:
            return {k: onehot_to_index(samples[k]).float() for k in samples}
        return samples

    def get_space(self, fixed):
        # Legacy: enumerate bit-vectors of length v_size[k]
        if self.domain_k is None:
            vals = []
            for k in self.v:
                if k in fixed:
                    vals.append([fixed[k]])
                else:
                    vals.append(list(range(2 ** self.v_size[k])))

            space = []
            for val_item in itertools.product(*vals):
                val_dict = dict()
                for i, k in enumerate(self.v):
                    val_dict[k] = self.dec_to_bin(T.as_tensor(val_item[i]), self.v_size[k])
                space.append(val_dict)
            return space

        # Categorical: enumerate one-hot vectors of size K
        K = int(self.domain_k)
        vals = []
        for k in self.v:
            if k in fixed:
                vals.append([fixed[k]])
            else:
                vals.append(list(range(K)))

        space = []
        for val_item in itertools.product(*vals):
            row = {}
            for i, k in enumerate(self.v):
                if k in fixed:
                    v = fixed[k]
                    # accept internal index or external one-hot
                    if T.is_tensor(v) and v.dim() == 2 and v.shape[1] == K:
                        row[k] = v.float()
                    else:
                        row[k] = index_to_onehot(T.as_tensor(v).view(1, 1), K).squeeze(0)
                else:
                    row[k] = index_to_onehot(T.as_tensor(val_item[i]).view(1, 1), K).squeeze(0)
            space.append(row)
        return space

    def likelihood(self, v_vals, u=None, skip=set(), mc_size=1):
        assert not skip.difference(self.v)

        if u is None:
            u = self.pu.sample(mc_size)
        else:
            mc_size = u[next(iter(u))].shape[0]

        expanded_vals = dict()

        if self.domain_k is None:
            # legacy behavior
            for (k, v) in v_vals.items():
                if not T.is_tensor(v) or len(v.shape) == 1:
                    expanded_vals[k] = expand_do(v, mc_size).float()
                else:
                    expanded_vals[k] = v.to(self.device_param)
        else:
            # categorical behavior: ensure each observed variable is one-hot [mc_size,K]
            K = int(self.domain_k)
            for (k, v) in v_vals.items():
                if not T.is_tensor(v):
                    v = T.as_tensor(v)

                # one-hot vector for a single sample: [K] -> [mc_size,K]
                if v.dim() == 1 and v.shape[0] == K:
                    expanded_vals[k] = expand_do(v.view(1, K), mc_size).to(self.device_param).float()
                    continue

                # already [mc,K]
                if v.dim() == 2 and v.shape[1] == K:
                    # if it's a single row, expand it
                    if v.shape[0] == 1 and mc_size != 1:
                        expanded_vals[k] = expand_do(v, mc_size).to(self.device_param).float()
                    else:
                        expanded_vals[k] = v.to(self.device_param).float()
                    continue

                # index [mc,1] or scalar
                if v.dim() == 0:
                    v = v.view(1, 1)
                elif v.dim() == 1:
                    v = v.view(-1, 1)

                if v.dim() == 2 and v.shape[1] == 1:
                    if v.shape[0] == 1:
                        v = expand_do(v, mc_size)
                    expanded_vals[k] = index_to_onehot(v.to(self.device_param), K)
                else:
                    # assume already broadcastable
                    expanded_vals[k] = v.to(self.device_param)

        log_pv = T.zeros(mc_size).to(self.device_param)
        for k in self.v:
            if k not in skip:
                log_pv += self.f[k](expanded_vals, u, expanded_vals[k])
        averaged_log_pv = T.logsumexp(log_pv, dim=0) - np.log(mc_size)
        return averaged_log_pv

    def sample(self, n=None, u=None, do={}, select=None):
        assert not set(do.keys()).difference(self.v)
        assert (n is None) != (u is None)

        if self.domain_k is None:
            for k in do:
                do[k] = do[k].to(self.device_param)
        else:
            # categorical: accept do-values as external index (scalar / [n,1]) or one-hot ([n,K])
            K = int(self.domain_k)
            do_oh = {}
            for k, v in do.items():
                if T.is_tensor(v) and v.dim() == 2 and v.shape[1] == K:
                    do_oh[k] = v.to(self.device_param).float()
                else:
                    # allow scalar, [n], [n,1]
                    if T.is_tensor(v):
                        v_t = v.to(self.device_param.device)
                    else:
                        v_t = T.as_tensor(v).to(self.device_param.device)
                    do_oh[k] = index_to_onehot(v_t, K)
            do = do_oh

        if u is None:
            u = self.pu.sample(n)
        if select is None:
            select = self.v
        v = {}
        remaining = set(select)
        for k in self.v:
            v[k] = do[k] if k in do else self.f[k](v, u, n=n)
            remaining.discard(k)
            if not remaining:
                break
        return {k: v[k] for k in select}

    def compute_ctf(self, query: CTF, n=1000000, u=None, get_prob=True, evaluating=False):
        if evaluating:
            return super().compute_ctf(query, n=n, u=u, get_prob=get_prob, evaluating=evaluating)

        if len(query.cond_term_set) > 0:
            cond_ctf = CTF(query.cond_term_set)
            full_ctf = CTF(query.term_set.union(query.cond_term_set))
            log_p_full = self.compute_ctf(full_ctf, n=n, u=u, get_prob=get_prob, evaluating=evaluating)
            log_p_cond = self.compute_ctf(cond_ctf, n=n, u=u, get_prob=get_prob, evaluating=evaluating)
            return log_p_full - log_p_cond

        u = self.pu.sample(n)
        log_prob = 0
        for term in query.term_set:
            fixed_vars = dict()
            do_vars = set()
            nested_vars = dict()
            for (k, v) in term.do_vals.items():
                if k == "nested":
                    nested_vars.update(self.compute_ctf(v, u=u, get_prob=False, evaluating=True))
                else:
                    fixed_vars[k] = v
                    do_vars.add(k)

            fixed_vars.update(term.var_vals)

            space = self.get_space(fixed_vars)

            log_prob_vals = []
            for row in space:
                val = {k: v.byte().to(self.device_param) for (k, v) in row.items()}
                for (k, v) in nested_vars.items():
                    val[k] = v
                log_prob_val = self.likelihood(val, u=u, skip=do_vars)
                log_prob_vals.append(log_prob_val)
            log_prob += T.logsumexp(T.stack(log_prob_vals, dim=0), dim=0)

        return log_prob

    def dec_to_bin(self, x, bits):
        mask = 2 ** T.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
