from lifelines import CoxPHFitter
from typing import Optional
import numpy as np
import pandas as pd


class EventSpecificFitter:
    """Event specific fitter - abstract class which specifies the API needed for a fitter model that will be
    used in CompetingRisksModel

    Note
    ---------
    Example of implementation can be seen below in the CoxWrapper class, that is also used in the example in
    file "first_exmaple.ipynb"
    """

    def fit(self,
            df: pd.DataFrame, duration_col: Optional[str], event_col: Optional[str], weights_col: Optional[str],
            cluster_col: Optional[str], entry_col: str, **fitter_kwargs):
        raise NotImplementedError('subclasses must override fit!')

    def get_coefficients(self) -> np.ndarray:
        raise NotImplementedError('subclasses must override coefficients!')

    def get_unique_event_times(self) -> np.ndarray:
        raise NotImplementedError('subclasses must override unique_event_times!')

    def get_baseline_hazard(self) -> np.ndarray:
        raise NotImplementedError('subclasses must override baseline_hazard!')

    def get_baseline_cumulative_hazard(self) -> np.ndarray:
        raise NotImplementedError('subclasses must override baseline_cumulative_hazard!')

    def print_summary(self):
        raise NotImplementedError('subclasses must override print_summary!')


class CoxWrapper(EventSpecificFitter):
    def __init__(self):
        self._model = CoxPHFitter()

    def fit(self, df: pd.DataFrame, duration_col: Optional[str], event_col: Optional[str], weights_col: Optional[str],
            cluster_col: Optional[str], entry_col: str, **fitter_kwargs):
        self._model.fit(df=df, duration_col=duration_col, event_col=event_col, weights_col=weights_col,
                        cluster_col=cluster_col, entry_col=entry_col, **fitter_kwargs)

    def get_coefficients(self) -> np.ndarray:
        return self._model.params_.values

    def get_unique_event_times(self) -> np.ndarray:
        return self._model.baseline_hazard_.index.values

    def get_baseline_hazard(self) -> np.ndarray:
        return self._model.baseline_hazard_["baseline hazard"].values

    def get_baseline_cumulative_hazard(self) -> np.ndarray:
        return self._model.baseline_cumulative_hazard_["baseline cumulative hazard"].values

    def print_summary(self):
        self._model.print_summary()
