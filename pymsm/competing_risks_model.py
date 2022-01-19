# -- R source: https://github.com/JonathanSomer/covid-19-multi-state-model/blob/master/model/competing_risks_model.R --#

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Dict, List
from lifelines import CoxPHFitter
from pandas.api.types import is_numeric_dtype
from pymsm.utils import stepfunc


class CompetingRisksModel:
    def __init__(self) -> None:
        """
        Each element of "event_specific_models" will be a dictionary
        with the following keys:
        1. coefficients
        2. unique_event_times
        3. baseline_hazard
        4. cumulative_baseline_hazard_function
        """
        self.failure_types: List[int] = []
        self.event_specific_models: Dict[Dict] = {}

    def assert_valid_dataset(
        self,
        df: pd.DataFrame,
        duration_col: str = None,
        event_col: str = None,
        cluster_col: str = None,
        weights_col: str = None,
        entry_col: str = None,
    ) -> None:

        assert df[duration_col].count() == df[event_col].count()

        # t should be non-negative
        assert (df[duration_col] >= 0).any(), "duration column has negative values"

        # failure types should be integers from 0 to m, not necessarily consecutive
        assert df[event_col].dtypes == int, "event column needs to be of type int"
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

    def break_ties_by_adding_epsilon(
        self, t: np.ndarray, epsilon_min: float = 0.0, epsilon_max: float = 0.0001
    ) -> np.ndarray:
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
        **coxph_kwargs,
    ) -> CoxPHFitter:
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

        cox_model = CoxPHFitter()
        cox_model.fit(
            df=event_df,
            duration_col=duration_col,
            event_col=event_col,
            weights_col=weights_col,
            cluster_col=cluster_col,
            entry_col=entry_col,
            **coxph_kwargs,
        )

        if verbose >= 2:
            cox_model.print_summary()

        return cox_model

    def extract_necessary_attributes(
        self, cox_model: CoxPHFitter
    ) -> Dict[str, np.ndarray]:
        return {
            "coefficients": cox_model.params_.values,
            "unique_event_times": cox_model.baseline_hazard_.index.values,
            "baseline_hazard": cox_model.baseline_hazard_["baseline hazard"].values,
            "cumulative_baseline_hazard_function": cox_model.baseline_cumulative_hazard_[
                "baseline cumulative hazard"
            ].values,
        }

    def compute_cif_function(
        self, sample_covariates: np.ndarray, failure_type: int
    ) -> interp1d:
        cif_x = self.unique_event_times(failure_type)
        cif_y = np.cumsum(
            self.hazard_at_unique_event_times(sample_covariates, failure_type)
            * self.survival_function(cif_x, sample_covariates)
        )
        return stepfunc(cif_x, cif_y)

    def hazard_at_unique_event_times(
        self, sample_covariates: np.ndarray, failure_type: int
    ) -> np.ndarray:
        # the hazard is given by multiplying the baseline hazard (which has value per unique event time) by the partial hazard
        hazard = self.baseline_hazard(failure_type) * (
            self.partial_hazard(failure_type, sample_covariates)
        )
        assert len(hazard) == len(self.unique_event_times(failure_type))
        return hazard

    def cumulative_baseline_hazard(cox_model: CoxPHFitter) -> np.ndarray:
        return cox_model.baseline_cumulative_hazard_[
            "baseline cumulative hazard"
        ].values

    def cumulative_baseline_hazard_step_function(self, cox_model: CoxPHFitter):
        return stepfunc(
            cox_model.baseline_hazard_.index.values,
            self.cumulative_baseline_hazard(cox_model),
        )

    def baseline_hazard(self, failure_type: int) -> np.ndarray:
        """
        the baseline hazard is given as a non-paramateric function, whose values are given at the times of observed events
        the cumulative hazard is the sum of hazards at times of events, the hazards are then the diffs 
        
        """
        return self.event_specific_models[failure_type]["baseline_hazard"]

    def partial_hazard(
        self, failure_type: int, sample_covariates: np.ndarray
    ) -> np.ndarray:
        # simply e^x_dot_beta for the chosen failure type's coefficients
        coefs = self.event_specific_models[failure_type]["coefficients"]
        x_dot_beta = sample_covariates * coefs
        return np.exp(x_dot_beta)

    def unique_event_times(self, failure_type: int) -> np.ndarray:
        # uses a coxph function which returns unique times, regardless of the original fit which might have tied times.
        return self.event_specific_models[failure_type]["unique_event_times"]

    def survival_function(
        self, t: np.ndarray, sample_covariates: np.ndarray
    ) -> np.ndarray:
        # simply: exp( sum of cumulative hazards of all types )
        exponent = np.zeros_like(t)
        for type in self.failure_types:
            exponent = exponent - (
                self.event_specific_models[type]["cumulative_baseline_hazard_function"][
                    t
                ]
                * (self.partial_hazard(type, sample_covariates))
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
        """This method fits a cox proportional hazards model for each failure type, treating others as censoring events.
        Tied event times are dealt with by adding an epsilon to tied event times.

        :param df: A pandas DataFrame contaning all relevant columns, must have duration and event columns. Optional to have clester and weight columns. All other columns other than these 4 will be treated as covariate columns
        :type df: pd.DataFrame
        :param duration_col: the name of the column in DataFrame that contains the subjects’ lifetimes, defaults to "T"
        :type duration_col: str, optional
        :param event_col: the name of the column in DataFrame that contains the subjects’ death observation, defaults to "E"
        :type event_col: str, optional
        :param cluster_col: specifies what column has unique identifiers for clustering covariances. Using this forces the sandwich estimator (robust variance estimator) to be used, defaults to None
        :type cluster_col: str, optional
        :param weights_col: an optional column in the DataFrame, df, that denotes the weight per subject. This column is expelled and not used as a covariate, but as a weight in the final regression. Default weight is 1. This can be used for case-weights. For example, a weight of 2 means there were two subjects with identical observations. This can be used for sampling weights. In that case, use robust=True to get more accurate standard errors, defaults to None
        :type weights_col: str, optional
        :param entry_col: a column denoting when a subject entered the study, i.e. left-truncation, defaults to None
        :type entry_col: str, optional
        :param break_ties: Break event ties by adding epsilon, defaults to True
        :type break_ties: bool, optional
        :param epsilon_min: epsilon is added to events with identical times to break ties. these values should be chosen so that they do not change the order of the events, defaults to 0.0
        :type epsilon_min: float, optional
        :param epsilon_max: epsilon is added to events with identical times to break ties. these values should be chosen so that they do not change the order of the events, defaults to 0.0001
        :type epsilon_max: float, optional
        :param verbose: verbosity, defaults to 1
        :type verbose: int, optional
        """

        self.assert_valid_dataset(df, duration_col, event_col, cluster_col, weights_col)

        if break_ties:
            t = df[duration_col].copy()
            df[duration_col] = self.break_ties_by_adding_epsilon(
                t, epsilon_min, epsilon_max
            )

        failure_types = df[event_col].unique()
        failure_types = failure_types[failure_types > 0]
        print(failure_types)
        for event_of_interest in failure_types:

            cox_model = self.fit_event_specific_model(
                event_of_interest,
                df,
                duration_col,
                event_col,
                cluster_col,
                weights_col,
                entry_col,
                verbose,
            )
            self.event_specific_models[
                event_of_interest
            ] = self.extract_necessary_attributes(cox_model)

    def predict_CIF(
        self,
        predict_at_t: np.ndarray,
        sample_covariates: np.ndarray,
        failure_type: int,
        time_passed: float = 0,
    ) -> np.ndarray:
        """
        Description:
        ------------------------------------------------------------------------------------------------------------
        This method computes the failure-type-specific cumulative incidence function, given that 'time_passed' time
        has passed (default is 0)
        
        Arguments:
        ------------------------------------------------------------------------------------------------------------
        predict_at_t: numeric vector
          times at which the cif will be computed
        
        sample_covariates: numeric vector
          a numerical vector of same length as the covariate matrix the model was fit to.
        
        failure_type: integer
          integer corresponing to the failure type, as given when fitting the model
        
        time_passed: numeric
          compute the cif conditioned on the fact that this amount of time has already passed.
        
        Returns:
        ------------------------------------------------------------------------------------------------------------
        the predicted cumulative incidence values for the given sample_covariates at times predict_at_t.
        """
        # Obtain CIF step function (interp1d function)
        cif_function = self.compute_cif_function(sample_covariates, failure_type)
        # Predict at t
        predictions = cif_function(predict_at_t)

        # re-normalize the probability to account for the time passed
        if time_passed > 0:
            predictions = (
                predictions - cif_function(time_passed)
            ) / self.survival_function(time_passed, sample_covariates)

        return predictions


if __name__ == "__main__":
    pass

