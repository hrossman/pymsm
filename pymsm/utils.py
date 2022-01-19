import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def stepfunc(xs: np.ndarray, ys: np.ndarray) -> interp1d:
    xs = np.concatenate((np.array([-np.inf]), xs))
    ys = np.concatenate((np.array([0]), ys))
    return interp1d(xs, ys, kind="previous", fill_value=np.nan, bounds_error=False)


def plot_stackplot(ax=None):
    # TODO
    pass
