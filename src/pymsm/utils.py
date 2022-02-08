import numpy as np
import pandas as pd
from typing import List
from scipy.interpolate import interp1d
from sklearn.preprocessing import OneHotEncoder


def stepfunc(xs: np.ndarray, ys: np.ndarray) -> interp1d:
    xs = np.concatenate((np.array([-np.inf]), xs))
    ys = np.concatenate((np.array([0]), ys))
    return interp1d(xs, ys, kind="previous", fill_value=np.nan, bounds_error=False)


def get_categorical_columns(df: pd.DataFrame, cat_cols: List) -> pd.DataFrame:
    encoder = OneHotEncoder(drop="first", sparse=False)
    new_df = pd.DataFrame(encoder.fit_transform(df[cat_cols]), dtype=int)
    new_df.columns = encoder.get_feature_names(cat_cols)
    return new_df
