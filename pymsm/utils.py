import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from scipy.interpolate import interp1d


def stepfunc(xs: np.ndarray, ys: np.ndarray) -> interp1d:
    xs = np.concatenate((np.array([-np.inf]), xs))
    ys = np.concatenate((np.array([0]), ys))
    return interp1d(xs, ys, kind="previous", fill_value=np.nan, bounds_error=False)


def plot_stackplot(times:np.ndarray ,cifs:List, labels:List=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots() 
    ax.stackplot(times, *cifs, labels=labels)
    ax.legend(loc="upper left")
    ax.set_xlim(times[0], times[-1])
    plt.show()
    return ax
