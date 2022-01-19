import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def stepfunc(xs: np.ndarray, ys:np.ndarray) -> interp1d:
    """[summary]

    :param xs: [description]
    :type xs: np.ndarray
    :param ys: [description]
    :type ys: np.ndarray
    :return: [description]
    :rtype: interp1d
    """
    return interp1d(xs, ys, kind="previous", fill_value=0, bounds_error=False)

