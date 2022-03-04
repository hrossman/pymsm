from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from typing import Optional
from pymsm.utils import stepfunc
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
        """Fit the model for a specific given state
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
        raise NotImplementedError("subclasses must override fit!")

    def get_unique_event_times(self) -> np.ndarray:
        """
        Get unique event times
        Returns
        -------
        np.ndarray
            unique event times that were encountered when fitting the model
        """
        raise NotImplementedError("subclasses must override unique_event_times!")

    def get_hazard(self, sample_covariates: np.ndarray) -> np.ndarray:
        """
        Get hazard for an individual with sample covariates
        Parameters
        ----------
        sample_covariates: np.ndarray
            covariates of the individual to calculate hazard for

        Returns
        -------
        np.ndarray
            hazard values for a specific individual, at the unique event times that were encountered when fitting the
            model
        """
        raise NotImplementedError("subclasses must override get_hazard!")

    def get_cumulative_hazard(
        self, t: np.ndarray, sample_covariates: np.ndarray
    ) -> np.ndarray:
        """

        Parameters
        ----------
        t: np.ndarray
            times in which to get cumulative hazard in
        sample_covariates: np.ndarray
            individual covariates

        Returns
        -------
        np.ndarray
            cumulative hazard values for a specific individual, at the unique event times that were encountered
            when fitting the model
        """
        raise NotImplementedError("subclasses must override get_cumulative_hazard!")

    def print_summary(self):
        """
        Prints summary of the model
        """
        raise NotImplementedError("subclasses must override print_summary!")


class CoxWrapper(EventSpecificFitter):
    def __init__(self):
        self._model = CoxPHFitter()

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
        try:
            self._model.fit(
                df=df,
                duration_col=duration_col,
                event_col=event_col,
                weights_col=weights_col,
                cluster_col=cluster_col,
                entry_col=entry_col,
                **fitter_kwargs,
            )
        except ConvergenceError:
            print(
                "ERROR! Model did not converge. Number of transitions in the data might not be sufficient."
            )
            raise

    def _get_coefficients(self) -> np.ndarray:
        return self._model.params_.values

    def get_unique_event_times(self) -> np.ndarray:
        return self._model.baseline_hazard_.index.values

    def _partial_hazard(self, sample_covariates):
        coefs = self._get_coefficients()
        x_dot_beta = np.dot(sample_covariates, coefs)
        return np.exp(x_dot_beta)

    def get_hazard(self, sample_covariates) -> np.ndarray:
        # the hazard is given by multiplying the baseline hazard (which has value per unique event time) by the partial hazard
        partial_hazard = self._partial_hazard(sample_covariates)
        baseline_hazard = self._model.baseline_hazard_["baseline hazard"].values
        hazard = baseline_hazard * partial_hazard
        return hazard

    def get_cumulative_hazard(self, t, sample_covariates) -> np.ndarray:
        baseline_cumulative_hazard = self._model.baseline_cumulative_hazard_[
            "baseline cumulative hazard"
        ].values
        cumulative_baseline_hazard_stepfunc = stepfunc(
            self.get_unique_event_times(), baseline_cumulative_hazard
        )
        cumulative_baseline_hazard = cumulative_baseline_hazard_stepfunc(t)
        partial_hazard = self._partial_hazard(sample_covariates)
        return cumulative_baseline_hazard * partial_hazard

    def print_summary(self):
        self._model.print_summary()


class ManualCoxWrapper(EventSpecificFitter):
    """Cox model, but derived from manual entry of parameters and baseline hazard. No fit available

    Note
    ---------
    coefs is an array of cox coefficients, one per covariate. Can be a numpy array or pandas Series. baselin_hazard is a pandas Series with unique event times as index and baseline hazard as values.
    """

    def __init__(self, coefs: pd.Series, baseline_hazard: pd.Series):
        if isinstance(coefs, pd.Series):
            coefs = coefs.values
        self.coefs = coefs
        assert isinstance(baseline_hazard, pd.Series)
        self.baseline_hazard = baseline_hazard.values
        self.unique_event_times = baseline_hazard.index.values

    def fit(self):
        raise NotImplementedError()

    def get_coefficients(self) -> np.ndarray:
        return self.coefs

    def get_unique_event_times(self) -> np.ndarray:
        return self.unique_event_times

    def _partial_hazard(self, sample_covariates):
        coefs = self.get_coefficients()
        x_dot_beta = np.dot(sample_covariates, coefs)
        return np.exp(x_dot_beta)

    def get_hazard(self, sample_covariates) -> np.ndarray:
        # the hazard is given by multiplying the baseline hazard (which has value per unique event time) by the partial hazard
        partial_hazard = self._partial_hazard(sample_covariates)
        baseline_hazard = self.baseline_hazard
        hazard = baseline_hazard * partial_hazard
        return hazard

    def get_cumulative_hazard(self, t, sample_covariates) -> np.ndarray:
        baseline_cumulative_hazard = self.baseline_hazard.cumsum()
        cumulative_baseline_hazard_stepfunc = stepfunc(
            self.get_unique_event_times(), baseline_cumulative_hazard
        )
        cumulative_baseline_hazard = cumulative_baseline_hazard_stepfunc(t)
        partial_hazard = self._partial_hazard(sample_covariates)
        return cumulative_baseline_hazard * partial_hazard

    def print_summary(self):
        print("Manual cox model")
        print(f"Coefficients: {self.coefs}")
