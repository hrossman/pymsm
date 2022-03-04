from pymsm.event_specific_fitter import EventSpecificFitter
from typing import Optional
from pymsm.utils import stepfunc
import numpy as np
import pandas as pd

try:
    from sksurv.tree import SurvivalTree
except ImportError:
    raise ImportError("pip install scikit-survival")


class SurvivalTreeWrapper(EventSpecificFitter):
    def __init__(self):
        self._model = SurvivalTree()

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: Optional[str],
        event_col: Optional[str],
        weights_col: Optional[str],
        cluster_col: Optional[str],
        entry_col: str,
        **fitter_kwargs,
    ):
        # TODO - how to use cluster_col and event_col
        covariate_cols = [
            col
            for col in df.columns
            if col not in [duration_col, event_col, cluster_col, entry_col]
        ]
        X = df[covariate_cols].copy()
        y_df = df[[event_col, duration_col]].copy()
        y_df[event_col] = y_df[event_col].astype(bool)
        y = np.asarray(y_df.to_records(index=False))
        self._model.fit(X, y, sample_weight=weights_col)

    def get_unique_event_times(self) -> np.ndarray:
        return self._model.event_times_

    def get_hazard(self, sample_covariates) -> np.ndarray:
        cumulative_hazard = self.get_cumulative_hazard(
            self.get_unique_event_times(), sample_covariates
        )
        hazard_df = pd.DataFrame(cumulative_hazard).diff()
        return hazard_df.values.ravel()

    def get_cumulative_hazard(self, t, sample_covariates) -> np.ndarray:
        cumulative_hazard_at_times = self._model.predict_cumulative_hazard_function(
            sample_covariates.reshape(1, -1), return_array=True
        ).ravel()
        times = self.get_unique_event_times()
        cum_hazard_stepfunc = stepfunc(times, cumulative_hazard_at_times)
        return cum_hazard_stepfunc(t)

    def print_summary(self):
        # TODO - print some summary of the tree
        return
