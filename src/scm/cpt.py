import torch as T
import numpy as np
from src.scm.scm import SCM
from src.scm.distribution.distribution import Distribution


class CPTExogenousDistribution(Distribution):
    """用CPT(无父节点的边缘分布)来采样外生变量U。"""

    def __init__(self, u_vars, u_probs, seed=None):
        super().__init__(list(u_vars))
        self.u_probs = {k: np.asarray(u_probs[k], dtype=float).reshape(-1) for k in u_vars}
        # 归一化，防止输入概率有微小数值误差
        for k, p in self.u_probs.items():
            s = float(p.sum())
            if s <= 0:
                raise ValueError(f"Invalid probability table for exogenous var {k}: sum={s}")
            self.u_probs[k] = p / s
        self._gen = np.random.RandomState(seed=seed)

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device
        out = {}
        for U, p in self.u_probs.items():
            samp = self._gen.choice(len(p), size=(n,), p=p)
            out[U] = T.from_numpy(samp).long().to(device).view(n, 1)
        return out


class CPT(SCM):
    """
    CPT: A Structural Causal Model where each variable is defined by a conditional probability table (CPT).
    - 默认将“无父节点”的变量视为外生变量U，并用其概率表作为 pu 来采样。
    - 这些U不会再通过结构方程 f 生成（避免覆盖 pu.sample 的结果）。
    """

    @staticmethod
    def topological_sort(variables, parents):
        """对变量进行拓扑排序，保证每个变量的父节点都在它前面。"""
        from collections import defaultdict, deque
        in_degree = {v: 0 for v in variables}
        graph = defaultdict(list)
        for v in variables:
            for p in parents.get(v, []):
                graph[p].append(v)
                in_degree[v] += 1
        queue = deque([v for v in variables if in_degree[v] == 0])
        order = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for nei in graph[v]:
                in_degree[nei] -= 1
                if in_degree[nei] == 0:
                    queue.append(nei)
        if len(order) != len(variables):
            raise ValueError("Graph has a cycle or missing variables!")
        return order

    def convert_evaluation(self, samples):
        """评估阶段默认不输出外生变量(U*)，让列集合与多数 NCM 输出一致。"""
        if getattr(self, 'u_vars', None):
            u_set = set(self.u_vars)
            return {k: v for (k, v) in samples.items() if k not in u_set}
        return samples

    def _u_passthrough(self, U_name: str):
        def f_u(v, u):
            if U_name not in u:
                raise KeyError(f"Exogenous variable {U_name} not found in sampled u")
            return u[U_name]
        return f_u

    def __init__(self, variables, cpt_tables, parents, state_sizes, pu=None, seed=None):
        # 统一在类内部做拓扑排序，避免外部重复/漏掉
        variables = CPT.topological_sort(list(variables), parents)

        self.variables = variables
        self.cpt_tables = cpt_tables
        self.parents = parents
        self.state_sizes = state_sizes

        # 自动识别外生变量：无父节点
        self.u_vars = [v for v in variables if len(parents.get(v, [])) == 0]

        # 如果没有提供 pu，则用无父节点变量的概率表构建 pu
        if pu is None:
            u_probs = {u: self.cpt_tables[u] for u in self.u_vars}
            pu = CPTExogenousDistribution(self.u_vars, u_probs, seed=seed)

        # Build f dict for SCM：
        # - 外生变量：从 u 中直接取
        # - 内生变量：用 CPT 采样
        v_endogenous = [v for v in variables if v not in self.u_vars]
        f = {U: self._u_passthrough(U) for U in self.u_vars}
        f.update({var: self._make_cpt_func(var) for var in v_endogenous})

        super().__init__(variables, f, pu)

    def _make_cpt_func(self, var):
        cpt = self.cpt_tables[var]
        parents = self.parents.get(var, [])
        state_sizes = [self.state_sizes[p] for p in parents]
        var_states = self.state_sizes[var]
        def func(v, u):
            # v: dict of parent values (tensor, shape [n, 1])
            # returns: tensor of sampled values (shape [n, 1])
            if parents:
                # 推断 batch size n：优先用 u（pu.sample(n) 的 n 最可靠）
                if u is not None and len(u) > 0:
                    any_u = u[next(iter(u))]
                    n = int(any_u.shape[0]) if any_u.dim() > 0 else 1
                elif len(v) > 0:
                    any_v = next(iter(v.values()))
                    n = int(any_v.shape[0]) if any_v.dim() > 0 else 1
                else:
                    n = 1

                parent_tensors = []
                for p in parents:
                    pv = v[p]
                    # 若是 [n,k] 取第一列
                    if pv.dim() == 0:
                        pv = pv.view(1)
                    elif pv.dim() >= 2:
                        pv = pv[:, 0]
                    pv = pv.reshape(-1).long()

                    # 标量扩展到 batch
                    if pv.numel() == 1 and n > 1:
                        pv = pv.repeat(n)

                    if pv.numel() != n:
                        raise ValueError(
                            f"Parent {p} for var {var} has {pv.numel()} samples, expected {n}. "
                            f"Original shape={tuple(v[p].shape)}"
                        )

                    parent_tensors.append(pv)

                parent_vals = T.stack(parent_tensors, dim=1)  # [n, len(parents)]

                idx = T.zeros(n, dtype=T.long)
                for i in range(len(parents)):
                    idx = idx * int(state_sizes[i]) + parent_vals[:, i]

                cpt_2d = cpt.reshape(-1, var_states)
                probs = cpt_2d[idx]
            else:
                # 无父节点（通常为外生变量U，但这里保留兼容）
                if u is not None and len(u) > 0:
                    any_u = u[next(iter(u))]
                    n = int(any_u.shape[0]) if any_u.dim() > 0 else 1
                elif len(v) > 0:
                    any_v = next(iter(v.values()))
                    n = int(any_v.shape[0]) if any_v.dim() > 0 else 1
                else:
                    n = 1
                probs = np.tile(cpt, (n, 1)) if cpt.ndim == 1 else cpt.repeat(n, axis=0)

            sampled = []
            for row in probs:
                if np.isscalar(row) or isinstance(row, np.float64):
                    sampled.append(int(row))
                else:
                    sampled.append(np.random.choice(var_states, p=row))
            return T.tensor(sampled, dtype=T.long).unsqueeze(1)
        return func

import xml.etree.ElementTree as ET
import numpy as np

def parse_smile_cpt(xml_path):
    """
    读取 SMILE/GeNIe 导出的 XML CPT 文件，返回 CPT 类所需的参数
    返回: variables, cpt_tables, parents, state_sizes
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # 兼容 <smile> 或 <smile ...>
    nodes = root.find('nodes')
    variables = []
    cpt_tables = {}
    parents = {}
    state_sizes = {}

    for cpt in nodes.findall('cpt'):
        var = cpt.attrib['id']
        variables.append(var)
        # 解析状态
        states = [s.attrib['id'] for s in cpt.findall('state')]
        state_sizes[var] = len(states)
        # 解析父节点
        parent_elems = cpt.find('parents')
        if parent_elems is not None:
            parent_list = parent_elems.text.strip().split()
        else:
            parent_list = []
        parents[var] = parent_list
        # 解析概率表
        prob_text = cpt.find('probabilities').text.strip()
        probs = np.array([float(x) for x in prob_text.split()])
        # 计算 CPT 形状
        shape = [state_sizes[p] for p in parent_list] + [len(states)]
        if len(probs) != np.prod(shape):
            raise ValueError(f"CPT size mismatch for {var}: expected {np.prod(shape)}, got {len(probs)}")
        cpt_tables[var] = probs.reshape(shape)
    return variables, cpt_tables, parents, state_sizes

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cpt.py <smile_xml_file>")
        return
    xml_path = sys.argv[1]
    print(f"Parsing {xml_path} ...")
    variables, cpt_tables, parents, state_sizes = parse_smile_cpt(xml_path)

    # 保证拓扑顺序（也会在 CPT() 内部再次确保）
    variables = CPT.topological_sort(variables, parents)
    print("Variables (topo order):", variables)
    print("Parents:", parents)
    print("State sizes:", state_sizes)

    # 仅打印外生变量(U*)的概率表，用于检查是否按真实分布采样
    u_vars = [v for v in variables if len(parents.get(v, [])) == 0]
    if u_vars:
        print("\nExogenous variables (U*) CPT/prob tables:")
        for u in u_vars:
            tab = np.asarray(cpt_tables[u]).reshape(-1)
            s = float(tab.sum())
            if s > 0:
                tab = tab / s
            print(f"  {u}: {tab.tolist()}")

    model = CPT(variables, cpt_tables, parents, state_sizes)

    # 测试采样
    n = 500
    samples = model.sample(n=n)

    # -------- 仅做统计检查（避免打印超长样本列表） --------
    samples_np = {k: samples[k].detach().cpu().reshape(-1).numpy() for k in variables}

    def _marginal(k: str):
        xs = samples_np[k]
        K = int(state_sizes[k])
        cnt = np.bincount(xs.astype(int), minlength=K).astype(float)
        return (cnt / cnt.sum()).tolist() if cnt.sum() > 0 else cnt.tolist()

    print("\nEmpirical marginals (from samples):")
    for k in variables:
        print(f"  {k}: {_marginal(k)}")

    # R vs W：翻转率应接近 0.00740884
    if 'R' in samples_np and 'W' in samples_np:
        flip = float(np.mean(samples_np['R'] != samples_np['W']))
        print(f"\nCheck R vs W: flip_rate=mean(R!=W)={flip}")

    # W/X/Y：CPT 多为 1/0 的确定性表。这里打印“每种父配置下是否确定”的比例。
    def _deterministic_fraction(var: str) -> float:
        par = parents.get(var, [])
        tab = np.asarray(cpt_tables[var])
        # 形状: [|P1|, |P2|, ..., |var|]
        # 对每个父配置，看 max(prob) 是否接近 1
        last = tab.shape[-1]
        flat = tab.reshape(-1, last)
        mx = flat.max(axis=1)
        return float(np.mean(mx > 0.999999))  # 近似 one-hot

    for v in ['W', 'X', 'Y']:
        if v in cpt_tables and len(parents.get(v, [])) > 0:
            frac = _deterministic_fraction(v)
            print(f"Check {v} CPT determinism: fraction(max_prob>0.999999)={frac}")

    # 如仍需要打印所有样本，把下面注释取消
    # print(f"\nSampled {n} examples:")
    # for k in variables:
    #     out = samples[k].detach().cpu().reshape(-1).tolist()
    #     print(f"{k}: {out}")

if __name__ == "__main__":
    main()