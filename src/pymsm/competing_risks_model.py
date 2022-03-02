import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Dict, List, Optional
from pandas.api.types import is_numeric_dtype
from pymsm.utils import stepfunc
from pymsm.event_specific_fitter import EventSpecificFitter, CoxWrapper


class EventSpecificModel:
    """Event specific model, holding attributes needed for later calculations

    Args:
        failure_type (int, optional): failure_type. Defaults to None.
        model (EventSpecificFitter, optional): Model for the specific failuure_type. Defaults to None.

    Attributes:
        unique_event_times (np.ndarray, optional): Array of unique event times
    """

    def __init__(self, failure_type=None, model=None):
        self.failure_type: int = failure_type
        self.model: EventSpecificFitter = model
        self.unique_event_times: Optional[np.ndarray] = None

    def extract_necessary_attributes(self) -> None:
        """Extract relevant arrays from event specific cox model"""
        self.unique_event_times = self.model.get_unique_event_times()


class CompetingRisksModel:
    """This class implements fitting a Competing Risk model.

    Args:
        event_specific_fitter (EventSpecificFitter, optional): The specified EventSpecificFitter. Defaults to CoxWrapper.


    Example:
        ```py linenums="1"

            from pymsm.examples.crm_example_utils import create_test_data, stack_plot
            from pymsm.competing_risks_model import CompetingRisksModel
            from pymsm.event_specific_fitter import CoxWrapper
            crm = CompetingRisksModel(CoxWrapper)
            data = create_test_data(N=1000)
            crm.fit(df=data, duration_col='T', event_col='transition', cluster_col='id')
        ```
    Attributes:
        failure_types (list): The possible failure types of the model
        event_specific_models (dict): A dictionary containing an instance of EventSpecificModel for each failure type.
    """

    failure_types: List[int]
    event_specific_models: Dict[int, EventSpecificModel]

    def __init__(self, event_specific_fitter: EventSpecificFitter = CoxWrapper):
        self.failure_types = []
        self.event_specific_models = {}
        self.event_specific_fitter = event_specific_fitter

    @staticmethod
    def assert_valid_dataset(
        df: pd.DataFrame,
        duration_col: str = None,
        event_col: str = None,
        cluster_col: str = None,
        weights_col: str = None,
        entry_col: str = None,
    ) -> None:
        """Checks if a dataframe is valid for fitting a competing risks model

        Args:
            df (pd.DataFrame): A pandas DataFrame contaning all relevant columns, must have duration and event columns. Optional to have clester and weight columns. All other columns other than these 4 will be treated as covariate columns.
            duration_col (str, optional): the name of the column in DataFrame that contains the subjects lifetimes, defaults to "T". Defaults to None.
            event_col (str, optional): the name of the column in DataFrame that contains the subjects death observation, defaults to "E". Defaults to None.
            cluster_col (str, optional): specifies what column has unique identifiers for clustering covariances. Using this forces the sandwich estimator (robust variance estimator) to be used, defaults to None. Defaults to None.
            weights_col (str, optional):  an optional column in the DataFrame, df, that denotes the weight per subject. This column is expelled and not used as a covariate, but as a weight in the final regression. Default weight is 1. This can be used for case-weights. For example, a weight of 2 means there were two subjects with identical observations. This can be used for sampling weights. In that case, use robust=True to get more accurate standard errors. Defaults to None.
            entry_col (str, optional): a column denoting when a subject entered the study, i.e. left-truncation. Defaults to None.
        """

        assert df[duration_col].count() == df[event_col].count()

        # t should be non-negative
        assert (df[duration_col] >= 0).any(), "duration column has negative values"

        # failure types should be integers from 0 to m, not necessarily consecutive
        assert pd.api.types.is_integer_dtype(
            df[event_col].dtypes
        ), "event column needs to be of type int"
        assert (
            df[event_col].min() >= 0
        ), "Failure types need to zero or positive integers"

        # covariates should all be numerical
        for col in df.columns:
            if col not in [
                duration_col,
                event_col,
                cluster_col,
                weights_col,
                entry_col,
            ]:
                assert is_numeric_dtype(
                    df[col]
                ), f"Non-numeric values found in {col} column"

    @staticmethod
    def break_ties_by_adding_epsilon(
        t: np.ndarray, epsilon_min: float = 0.0, epsilon_max: float = 0.0001
    ) -> np.ndarray:
        """Breaks ties in event times by adding a samll random number

        Parameters
        ----------
        t : np.ndarray
            array of event times
        epsilon_min : float, optional
            minimum value, by default 0.0
        epsilon_max : float, optional
            maximum value, by default 0.0001

        Returns
        -------
        np.ndarray
            array of event times with ties broken
        """
        np.random.seed(42)
        _, inverse, count = np.unique(
            t, return_inverse=True, return_counts=True, axis=0
        )
        non_unique_times_idx = np.where(count[inverse] > 1)[
            0
        ]  # find all indices where counts > 1

        # Add epsilon to all non-unique events
        eps = np.random.uniform(
            low=epsilon_min, high=epsilon_max, size=len(non_unique_times_idx)
        )
        t_new = t.copy().astype(float)
        np.add.at(t_new, non_unique_times_idx, eps)
        # Leave time zero as it was
        t_new[0] = t[0]
        return t_new

    def fit_event_specific_model(
        self,
        event_of_interest: int,
        df: pd.DataFrame,
        duration_col: str = "T",
        event_col: str = "E",
        cluster_col: str = None,
        weights_col: str = None,
        entry_col: str = None,
        verbose: int = 1,
        **fitter_kwargs,
    ) -> EventSpecificFitter:
        """Fits a the model in EventSpecificFitter for a specific event of interest. Applies censoring to other events

        Parameters
        ----------
        event_of_interest : int
            The event which is to be fitted
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
        verbose : int, optional
            verbosity, by default 1

        Returns
        -------
        EventSpecificFitter
             fitted model for a specific event of interest
        """

        # Treat all 'failure_types' except 'event_of_interest' as censoring events
        is_event = df[event_col] == event_of_interest

        # Apply censoring to all but event_of_interest
        event_df = df.copy()
        event_df.loc[~is_event, event_col] = 0
        event_df.loc[is_event, event_col] = 1

        if verbose >= 1:
            print(
                f">>> Fitting Transition to State: {event_of_interest}, n events: {np.sum(is_event)}"
            )

        event_fitter = self.event_specific_fitter()
        event_fitter.fit(
            df=event_df,
            duration_col=duration_col,
            event_col=event_col,
            weights_col=weights_col,
            cluster_col=cluster_col,
            entry_col=entry_col,
            **fitter_kwargs,
        )

        if verbose >= 2:
            event_fitter.print_summary()
        return event_fitter

    def _compute_cif_function(
        self, sample_covariates: np.ndarray, failure_type: int
    ) -> interp1d:
        """Computes the Cumulative Incidince (step) Function for a given failure type and set of covariates

        Parameters
        ----------
        sample_covariates : np.ndarray
            covariates used to build CIF
        failure_type : int
            failure type of interest

        Returns
        -------
        interp1d
            interpolation step function for the CIF
        """
        cif_x = self.unique_event_times(failure_type)
        hazard = self.hazard_at_unique_event_times(sample_covariates, failure_type)
        survival_func = self.survival_function(cif_x, sample_covariates)
        cif_y = np.cumsum(hazard * survival_func)
        return stepfunc(cif_x, cif_y)

    def hazard_at_unique_event_times(
        self, sample_covariates: np.ndarray, failure_type: int
    ) -> np.ndarray:
        """Hazard at unique event times

        Parameters
        ----------
        sample_covariates : np.ndarray
            covariates
        failure_type : int
            failure type of interest

        Returns
        -------
        np.ndarray
            hazard at unique event times
        """
        hazard = self.event_specific_models[failure_type].model.get_hazard(
            sample_covariates
        )
        assert len(hazard) == len(self.unique_event_times(failure_type))
        return hazard

    def unique_event_times(self, failure_type: int) -> np.ndarray:
        """Fetch unique event times for specific failure type

        Parameters
        ----------
        failure_type : int
            failure type of interest

        Returns
        -------
        np.ndarray
            unique event times for specific failure type
        """
        return self.event_specific_models[failure_type].unique_event_times

    def survival_function(
        self, t: np.ndarray, sample_covariates: np.ndarray
    ) -> np.ndarray:
        """Calculate survival function for a specific set of covariates at times t

        Parameters
        ----------
        t : np.ndarray
            times in which to calculate survival function
        sample_covariates : np.ndarray
            covariates

        Returns
        -------
        np.ndarray
            survival function for a specific set of covariates at times t
        """
        # simply: exp( sum of cumulative hazards of all types )
        exponent = np.zeros_like(t)
        for type in self.failure_types:
            exponent = exponent - (
                self.event_specific_models[type].model.get_cumulative_hazard(
                    t, sample_covariates
                )
            )
        survival_function_at_t = np.exp(exponent)
        assert len(survival_function_at_t) == len(t)
        return survival_function_at_t

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "T",
        event_col: str = "E",
        cluster_col: str = None,
        weights_col: str = None,
        entry_col: str = None,
        break_ties: bool = True,
        epsilon_min: float = 0.0,
        epsilon_max: float = 0.0001,
        verbose: int = 1,
    ) -> None:
        """Fit a cox proportional hazards model for each failure type, treating others as censoring events. Tied event times are dealt with by adding an epsilon to tied event times.

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
        break_ties : bool, optional
            Break event ties by adding epsilon, by default True
        epsilon_min : float, optional
            minimum value, by default 0.0
        epsilon_max : float, optional
            maximum value, by default 0.0001
        verbose : int, optional
            verbosity, by default 1
        """

        self.assert_valid_dataset(df, duration_col, event_col, cluster_col, weights_col)

        if break_ties:
            t = df[duration_col].copy()
            df.loc[:, duration_col] = self.break_ties_by_adding_epsilon(
                t, epsilon_min, epsilon_max
            )

        failure_types = df[event_col].unique()
        failure_types = failure_types[
            failure_types > 0
        ]  # Do not include censoring as failure_type

        # Save failure type
        self.failure_types = failure_types

        for event_of_interest in failure_types:
            # Fit cox model for specific event
            event_fitted_model = self.fit_event_specific_model(
                event_of_interest,
                df,
                duration_col,
                event_col,
                cluster_col,
                weights_col,
                entry_col,
                verbose,
            )
            # Save as instance of event_specific_model
            event_specific_model = EventSpecificModel(
                failure_type=event_of_interest, model=event_fitted_model
            )
            event_specific_model.extract_necessary_attributes()
            # Add to event_specific_models dictionary
            self.event_specific_models[event_of_interest] = event_specific_model

    def predict_CIF(
        self,
        predict_at_t: np.ndarray,
        sample_covariates: np.ndarray,
        failure_type: int,
        time_passed: float = 0,
    ) -> np.ndarray:
        """computes the failure-type-specific cumulative incidence function, given that 'time_passed' time  has passed (default is 0)

        Parameters
        ----------
        predict_at_t : np.ndarray
            times at which the cif will be computed
        sample_covariates : np.ndarray
            covariates array of same length as the covariate matrix the model was fit to
        failure_type : int
            integer corresponing to the failure type, as given when fitting the model
        time_passed : float, optional
            compute the cif conditioned on the fact that this amount of time has already passed, by default 0

        Returns
        -------
        np.ndarray
            Returns the predicted cumulative incidence values for the given sample_covariates at times predict_at_t
        """

        # Obtain CIF step function (interp1d function)
        cif_function = self._compute_cif_function(sample_covariates, failure_type)
        # Predict at t
        predictions = cif_function(predict_at_t)

        # re-normalize the probability to account for the time passed
        if time_passed > 0:
            predictions = (
                predictions - cif_function(time_passed)
            ) / self.survival_function(np.array([time_passed]), sample_covariates)

        return predictions

    def print_summary(self):
        """Print summary of fitted models"""
        for event_of_interest in self.failure_types:
            print(f"Model for failure type {event_of_interest}:\n")
            self.event_specific_models[event_of_interest].model.print_summary()
