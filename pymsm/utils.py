import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def stepfunc(xs: np.ndarray, ys: np.ndarray) -> interp1d:
    return interp1d(xs, ys, kind="previous", fill_value=0, bounds_error=False)


def plot_stackplot(ax=None):
    # TODO
    pass
