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
        """ Fit the model for a specific given state
        Parameters
        ----------
        df : pd.DataFrame
            A pandas DataFrame contaning all relevant columns, must have duration and event columns. Optional to have clester and weight columns. All other columns other than these 4 will be treated as covariate columns.
        duration_col : str, optional
            the name of the column in DataFrame that contains the subjects lifetimes, defaults to "T", by default None
        event_col : str, optional
            the name of the column in DataFrame that contains the subjects death observation, defaults to "E", by default None
        cluster_col : str, optional
            specifies what column has unique identifiers for clustering covariances. Using this forces the sandwich estimator (robust variance estimator) to be used, defaults to None, by default None
        weights_col : str, optional
            an optional column in the DataFrame, df, that denotes the weight per subject. This column is expelled and not used as a covariate, but as a weight in the final regression. Default weight is 1. This can be used for case-weights. For example, a weight of 2 means there were two subjects with identical observations. This can be used for sampling weights. In that case, use robust=True to get more accurate standard errors, by default None
        entry_col : str, optional
            a column denoting when a subject entered the study, i.e. left-truncation, by default None

        """
        raise NotImplementedError('subclasses must override fit!')

    def get_coefficients(self) -> np.ndarray:
        """
        Get fitted model coefficients
        Returns
        -------
        np.ndarray
            fitted model coefficients
        """
        raise NotImplementedError('subclasses must override coefficients!')

    def get_unique_event_times(self) -> np.ndarray:
        """
        Get unique event times
        Returns
        -------
        np.ndarray
            unique event times that were encountered when fitting the model
        """
        raise NotImplementedError('subclasses must override unique_event_times!')

    def get_baseline_hazard(self) -> np.ndarray:
        """
        Get baseline hazard
        Returns
        -------
        np.ndarray
            baseline hazard from the fitted model
        """
        raise NotImplementedError('subclasses must override baseline_hazard!')

    def get_baseline_cumulative_hazard(self) -> np.ndarray:
        """
        Get baseline cumulative hazard
        Returns
        -------
        np.ndarray
            baseline cumulative hazard of the fitted model
        """
        raise NotImplementedError('subclasses must override baseline_cumulative_hazard!')

    def print_summary(self):
        """
        Prints summary of the model
        """
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
