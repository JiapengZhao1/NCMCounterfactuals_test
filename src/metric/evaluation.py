import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch as T

from src.scm.scm import expand_do
from src.ds import CTF, CTFTerm


def eval_query(m, ctf, n=1000000):
    if isinstance(ctf, CTF):
        return m.compute_ctf(ctf, n=n, evaluating=True)
    else:
        return ctf_sum(m, ctf, n=n)


def ctf_sum(m, ctf_list, n=1000000):
    total = 0
    for ctf, sign in ctf_list:
        total += sign * m.compute_ctf(ctf, n=n, evaluating=True)
    return total


def probability_table(m=None, n=1000000, do={}, dat=None):
    assert m is not None or dat is not None

    if dat is None:
        dat = m(n, do=do, evaluating=True)

    cols = dict()
    for v in sorted(dat):
        result = dat[v].detach().numpy()
        if result.shape[1] == 1:  # Single-dimensional variable
            cols[v] = np.squeeze(result)  # Use the variable name directly
        else:  # Multi-dimensional variable
            for i in range(result.shape[1]):
                cols["{}{}".format(v, i)] = np.squeeze(result[:, i])

    df = pd.DataFrame(cols)
    return (df.groupby(list(df.columns))
            .apply(lambda x: len(x) / len(df))
            .rename('P(V)').reset_index()
            [[*df.columns, 'P(V)']])


def kl(truth, ncm, n=1000000, do={}, true_pv=None):
    m_table = probability_table(ncm, n=n, do=do)
    t_table = true_pv if true_pv is not None else probability_table(truth, n=n, do=do)
    cols = list(t_table.columns[:-1])
    joined_table = t_table.merge(m_table, how='left', on=cols, suffixes=['_t', '_m']).fillna(0.0000001)
    p_t = joined_table['P(V)_t']
    p_m = joined_table['P(V)_m']
    return (p_t * (np.log(p_t) - np.log(p_m))).sum()


def supremum_norm(truth, ncm, n=1000000, do={}, true_pv=None):
    m_table = probability_table(ncm, n=n, do=do)
    t_table = true_pv if true_pv is not None else probability_table(truth, n=n)
    cols = list(t_table.columns[:-1])
    joined_table = t_table.merge(m_table, how='outer', on=cols, suffixes=['_t', '_m']).fillna(0.0)
    p_t = joined_table['P(V)_t']
    p_m = joined_table['P(V)_m']
    return (p_m - p_t).abs().max()


def serialize_do(do_set):
    if len(do_set) == 0:
        return "P(V)"
    name = "P(V | do("
    for (k, v) in do_set.items():
        name += "{}={}, ".format(k, v)
    name = name[:-2] + "))"
    return name


def serialize_query(query):
    if isinstance(query, CTF):
        if query.name is not None:
            return query.name
    else:
        q = query[0][0]
        if q.name is not None:
            return q.name
    return str(query)


def all_metrics(truth, ncm, dat_dos, dat_sets, n=1000000, stored=None, query_track=None, include_sup=False):
    true_ps = dict()
    dat_ps = dict()
    m = dict()
    m["total_true_KL"] = 0
    m["total_dat_KL"] = 0
    if include_sup:
        m["total_true_supnorm"] = 0
        m["total_dat_supnorm"] = 0
    for i, do_set in enumerate(dat_dos):
        name = serialize_do(do_set)
        true_name = "true_{}".format(name)
        dat_name = "dat_{}".format(name)
        expanded_do_dat = {k: expand_do(v, n) for (k, v) in do_set.items()}
        if stored is None or true_name not in stored:
            true_ps[name] = None
        else:
            true_ps[name] = stored[true_name]
        if stored is None or dat_name not in stored:
            dat_ps[name] = probability_table(m=None, n=n, do=expanded_do_dat, dat=dat_sets[i])
        else:
            dat_ps[name] = stored[dat_name]

        m["true_KL_{}".format(name)] = kl(truth, ncm, n=n, do=expanded_do_dat, true_pv=true_ps[name])
        m["dat_KL_{}".format(name)] = kl(truth, ncm, n=n, do=expanded_do_dat, true_pv=dat_ps[name])
        m["total_true_KL"] += m["true_KL_{}".format(name)]
        m["total_dat_KL"] += m["dat_KL_{}".format(name)]
        if include_sup:
            m["true_supnorm_{}".format(name)] = supremum_norm(truth, ncm, n=n, do=expanded_do_dat,
                                                              true_pv=true_ps[name])
            m["dat_supnorm_{}".format(name)] = supremum_norm(truth, ncm, n=n, do=expanded_do_dat,
                                                             true_pv=dat_ps[name])
            m["total_true_supnorm"] += m["true_supnorm_{}".format(name)]
            m["total_dat_supnorm"] += m["dat_supnorm_{}".format(name)]
        
    if query_track is not None and query_track is not "avg_error":
        true_q = 'true_{}'.format(serialize_query(query_track))
        m[true_q] = eval_query(truth, query_track, n) if stored is None or true_q not in stored else stored[true_q]
        ncm_q = 'ncm_{}'.format(serialize_query(query_track))
        m[ncm_q] = eval_query(ncm, query_track, n)
        err_q = 'err_ncm_{}'.format(serialize_query(query_track))
        m[err_q] = m[true_q] - m[ncm_q]

    return m


def all_metrics_minmax(truth, ncm_min, ncm_max, dat_dos, dat_sets, n=1000000, stored=None, query_track=None):
    true_ps = dict()
    dat_ps = dict()
    m = dict()
    m["min_total_true_KL"] = 0
    m["min_total_dat_KL"] = 0
    m["min_total_true_supnorm"] = 0
    m["min_total_dat_supnorm"] = 0
    m["max_total_true_KL"] = 0
    m["max_total_dat_KL"] = 0
    m["max_total_true_supnorm"] = 0
    m["max_total_dat_supnorm"] = 0
    for i, do_set in enumerate(dat_dos):
        name = serialize_do(do_set)
        true_name = "true_{}".format(name)
        dat_name = "dat_{}".format(name)
        expanded_do_dat = {k: expand_do(v, n) for (k, v) in do_set.items()}
        if stored is None or true_name not in stored:
            true_ps[name] = None
        else:
            true_ps[name] = stored[true_name]
        if stored is None or dat_name not in stored:
            dat_ps[name] = probability_table(m=None, n=n, do=expanded_do_dat, dat=dat_sets[i])
        else:
            dat_ps[name] = stored[dat_name]

        m["min_true_KL_{}".format(name)] = kl(truth, ncm_min, n=n, do=expanded_do_dat, true_pv=true_ps[name])
        m["max_true_KL_{}".format(name)] = kl(truth, ncm_max, n=n, do=expanded_do_dat, true_pv=true_ps[name])
        m["min_dat_KL_{}".format(name)] = kl(truth, ncm_min, n=n, do=expanded_do_dat, true_pv=dat_ps[name])
        m["max_dat_KL_{}".format(name)] = kl(truth, ncm_max, n=n, do=expanded_do_dat, true_pv=dat_ps[name])
        m["min_true_supnorm_{}".format(name)] = supremum_norm(truth, ncm_min, n=n, do=expanded_do_dat,
                                                              true_pv=true_ps[name])
        m["max_true_supnorm_{}".format(name)] = supremum_norm(truth, ncm_max, n=n, do=expanded_do_dat,
                                                              true_pv=true_ps[name])
        m["min_dat_supnorm_{}".format(name)] = supremum_norm(truth, ncm_min, n=n, do=expanded_do_dat,
                                                             true_pv=dat_ps[name])
        m["max_dat_supnorm_{}".format(name)] = supremum_norm(truth, ncm_max, n=n, do=expanded_do_dat,
                                                             true_pv=dat_ps[name])

        m["min_total_true_KL"] += m["min_true_KL_{}".format(name)]
        m["min_total_dat_KL"] += m["min_dat_KL_{}".format(name)]
        m["min_total_true_supnorm"] += m["min_true_supnorm_{}".format(name)]
        m["min_total_dat_supnorm"] += m["min_dat_supnorm_{}".format(name)]
        m["max_total_true_KL"] += m["max_true_KL_{}".format(name)]
        m["max_total_dat_KL"] += m["max_dat_KL_{}".format(name)]
        m["max_total_true_supnorm"] += m["max_true_supnorm_{}".format(name)]
        m["max_total_dat_supnorm"] += m["max_dat_supnorm_{}".format(name)]

    if query_track is not None:
        true_q = 'true_{}'.format(serialize_query(query_track))
        m[true_q] = eval_query(truth, query_track, n) if stored is None or true_q not in stored else stored[true_q]
        min_ncm_q = 'min_ncm_{}'.format(serialize_query(query_track))
        max_ncm_q = 'max_ncm_{}'.format(serialize_query(query_track))
        m[min_ncm_q] = eval_query(ncm_min, query_track, n)
        m[max_ncm_q] = eval_query(ncm_max, query_track, n)
        min_err_q = 'min_err_ncm_{}'.format(serialize_query(query_track))
        max_err_q = 'max_err_ncm_{}'.format(serialize_query(query_track))
        m[min_err_q] = m[true_q] - m[min_ncm_q]
        m[max_err_q] = m[true_q] - m[max_ncm_q]
        minmax_gap = 'minmax_{}_gap'.format(serialize_query(query_track))
        m[minmax_gap] = m[max_ncm_q] - m[min_ncm_q]
    return m


def naive_metrics(truth, gan, do, n=1000000, dat_set=None, stored=None, query_track=None):

    m = dict()
    true_pv = stored['true_pv']
    dat_pv = stored['dat_pv']
    gan_pv = naive_probability_table(m=gan, n=n)
    true_q = 'true_{}'.format(serialize_query(query_track))
    m[true_q] = stored[true_q]
    # m[true_q] = eval_query(truth, query_track, n) if stored is None or true_q not in stored else stored[true_q]

    m["true_KL"] = naive_kl(true_pv, gan_pv)
    m["dat_KL"] = naive_kl(dat_pv, gan_pv)
    m["gan_naive_query"] = naive_query(gan, do=do, n=n)
    m["dat_naive_query"] = naive_query(do=do, dat=dat_set)

    err_q = 'err_gan_{}'.format(serialize_query(query_track))
    m[err_q] = m[true_q] - m["gan_naive_query"]
    return m


def naive_query(gan=None, do={}, dat=None, n=1000000):
    if dat is None:
        dat = gan(n, evaluating=True)
    if len(do) == 0:
        p_y1x1 = ((dat['X'] == 1) & (dat['Y'] == 1)).float().mean()
        p_y1x0 = ((dat['X'] == 0) & (dat['Y'] == 1)).float().mean()
        p_x1 = (dat['X'] == 1).float().mean()
        p_x0 = (dat['X'] == 0).float().mean()
        res = (p_y1x1 / (p_x1 + 1e-10) - p_y1x0 / (p_x0 + 1e-10)).item()
        return res
    else:
        p_y1 = (dat['Y'] == 1).float().mean().item()
        return p_y1


def naive_probability_table(m=None, dat_set=None, n=1000000):
    if dat_set is None:
        dat_set = m(n, evaluating=True)

    dat = {k: v.cpu() for (k, v) in dat_set.items()}
    return probability_table(n=n, dat=dat)


def naive_kl(t_table, m_table):
    cols = list(t_table.columns[:-1])
    joined_table = t_table.merge(m_table, how='left', on=cols, suffixes=['_t', '_m']).fillna(0.0000001)
    p_t = joined_table['P(V)_t']
    p_m = joined_table['P(V)_m']
    return (p_t * (np.log(p_t) - np.log(p_m))).sum()


def compute_average_errors(truth, estimated, n=1000000, dat_dos=[], true_pv=None):
    """
        Compute the average errors for the query P(Y | do(X)).

        Args:
            truth: The true model (e.g., SCM or CTM).
            estimated: The estimated model (e.g., GAN_NCM or another SCM).
            n: Number of samples to generate.
            dat_dos: List of dictionaries specifying interventions (e.g., [{"X": 0}, {"X": 1}]).

        Returns:
            dict: A dictionary containing the average errors for each combination of X and Y.
        """
    errors = {}

    # Convert dat_sets[0] (true_pv) to probability_table format if provided
    if true_pv is not None:
        true_pv = convert_dat_sets_to_probability_table(true_pv)
        #print(true_pv)

    for i, do_set in enumerate([{"X": 0}, {"X": 1}]):
        # Expand the do_set for the given number of samples
        expanded_do_dat = {k: expand_do(v, n) for (k, v) in do_set.items()}

        # Generate probability tables for the true and estimated models
        #true_table = probability_table(m=truth, n=n, do=expanded_do_dat)
        true_table = true_pv if true_pv is not None else probability_table(truth, n=n, do=expanded_do_dat)
        estimated_table = probability_table(m=estimated, n=n, do=expanded_do_dat)
        print(estimated_table)

        # Ensure Y is present
        y_column = None
        for col in true_table.columns:
            if col.startswith("Y"):
                y_column = col
                break

        if y_column is None:
            raise KeyError("The required column for 'Y' is missing in the probability table.")

        # Extract unique values of Y
        y_values = sorted(true_table[y_column].unique())

        # Extract the intervention variable dynamically
        intervention_var = list(do_set.keys())
        intervention_value = list(do_set.values())

        # Compute errors for each value of Y under the current do(X)
        for y in y_values:
            # Extract probabilities for the true model
            true_prob = true_table[true_table[y_column] == y]['P(V)'].sum()

            # Extract probabilities for the estimated model
            estimated_prob = estimated_table[estimated_table[y_column] == y]['P(V)'].sum()

            # Compute the absolute error
            error = abs(true_prob - estimated_prob)
            errors[f"P(Y={y} | do({intervention_var}={intervention_value}))"] = error

    # Compute the average error across all interventions
    avg_error = sum(errors.values()) / len(errors)
    errors["avg_error"] = avg_error
    return errors


def convert_dat_sets_to_probability_table(dat_sets):
    """
    Convert dat_sets format to probability_table format.

    Args:
        dat_sets (dict): A dictionary of tensors representing the dataset.

    Returns:
        pd.DataFrame: A DataFrame in the probability_table format with descriptive column names.
    """
    # Convert tensors to 1D arrays and create a DataFrame with variable names as column names
    df = pd.DataFrame({key: value.squeeze().tolist() for key, value in dat_sets.items()})

    # Group by unique combinations of variable values and compute probabilities
    probability_table = (
        df.groupby(list(df.columns))  # Group by all columns (unique combinations of variable values)
        .size()  # Count occurrences of each combination
        .div(len(df))  # Divide by total rows to compute probabilities
        .rename('P(V)')  # Rename the computed column to 'P(V)'
        .reset_index()  # Reset the index to make it a flat DataFrame
    )

    # Rename columns to include variable names explicitly
    probability_table.columns = [f"{col}" if col != 'P(V)' else col for col in probability_table.columns]

    return probability_table