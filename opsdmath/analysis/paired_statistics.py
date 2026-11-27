"""Paired question and cluster bootstrap routines used by the research studies."""

import random
from statistics import mean

import numpy as np


def interval(values):
    if not len(values):
        return dict(n=0, pp=None, interval95_pp=None)
    rng = np.random.default_rng(84081)
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return dict(n=len(values), pp=float(values.mean()*100), interval95_pp=(np.quantile(draws, [.025, .975])*100).tolist())


def uncertainty(deltas):
    rng = random.Random(55)
    n = len(deltas)
    draws = sorted(mean(rng.choices(deltas, k=n)) for _ in range(10000))
    return [draws[250], draws[9749]]


def cluster_interval(diffs):
    rng = random.Random(229)
    families = list(diffs)
    samples = sorted(mean(value for family in rng.choices(families, k=len(families))
                          for value in diffs[family]) for _ in range(5000))
    return [samples[125], samples[4874]]
