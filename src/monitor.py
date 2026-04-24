import numpy as np
import pandas as pd
from scipy import stats


def population_stability_index(expected, actual, buckets=10):
    # PSI calculation: compare distributions
    def _get_bins(series, bins):
        return np.histogram(series, bins=bins)[1]

    if np.issubdtype(expected.dtype, np.number) and np.issubdtype(actual.dtype, np.number):
        bins = _get_bins(expected, buckets)
        exp_perc, _ = np.histogram(expected, bins=bins)
        act_perc, _ = np.histogram(actual, bins=bins)
    else:
        # categorical: use value counts
        exp_counts = expected.value_counts(normalize=True)
        act_counts = actual.value_counts(normalize=True)
        categories = set(exp_counts.index) | set(act_counts.index)
        exp_perc = np.array([exp_counts.get(k, 0) for k in categories])
        act_perc = np.array([act_counts.get(k, 0) for k in categories])

    # convert to percentages
    exp_perc = exp_perc.astype(float) / np.sum(exp_perc)
    act_perc = act_perc.astype(float) / np.sum(act_perc)

    # avoid zeros
    act_perc = np.where(act_perc == 0, 1e-8, act_perc)
    exp_perc = np.where(exp_perc == 0, 1e-8, exp_perc)

    psi = np.sum((exp_perc - act_perc) * np.log(exp_perc / act_perc))
    return psi


def ks_test_series(a: pd.Series, b: pd.Series):
    # two-sample KS test for numeric distributions
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        stat, p = stats.ks_2samp(a, b)
        return stat, p
    return None
