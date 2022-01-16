# -- R source: https://github.com/JonathanSomer/covid-19-multi-state-model/blob/master/model/competing_risks_model.R --#

import numpy as np
from typing import List


class CompetingRisksModel:
    def __init__(
        self, failure_types: List = None, event_specific_models: List = None
    ) -> None:
        """
        Each element of "event_specific_models"is a list
        with the following attributes:
        1. coefficients
        2. unique_event_times
        3. baseline_hazard
        4. cumulative_baseline_hazard_function
        """
        pass

    def fit(
        self,
        t: np.ndarray,
        failure_types: List,
        covariates_X: List,
        sample_ids: List=None,
        t_start:float=None,
        break_ties: bool = True,
        sample_weights:np.ndarray=None,
        epsilon_min: float = 0.0,
        epsilon_max: float = 0.0001,
        verbose: int = 1,
    ):
        """
        Description:
        ------------------------------------------------------------------------------------------------------------
        This method fits a cox proportional hazards model for each failure type, treating others as censoring events.
        Tied event times are dealt with by adding an epsilon to tied event times.

        Arguments:
        ------------------------------------------------------------------------------------------------------------
        t : numeric vector
        A length n vector of positive times of events

        failure_types: numeric vector
        The event type corresponding to the time in vector t.
        Failure types are encoded as integers from 1 to m.
        Right-censoring events (the only kind of censoring supported) are encoded as 0.
        Thus, the failure type argument holds integers from 0 to m, where m is the number of distinct failure types

        covariates_X: numeric dataframe
        an n by #(covariates) numerical matrix
        All columns are used in the estimate.

        OPTIONAL:

        sample_ids:
        used inside the coxph model in order to identify subjects with repeating entries.

        t_start:
        A length n vector of positive start times, used in case of interval data.
        In that case: left=t_start, and right=t

        epsilon_min/max:
        epsilon is added to events with identical times to break ties.
        epsilon is sampled from a uniform distribution in the range (epsilon_min, epsilon_max)
        these values should be chosen so that they do not change the order of the events.
        """

        # TODO
        # assert_valid_dataset(t, failure_types, covariates_X)
    
        # if (break_ties) t = break_ties_by_adding_epsilon(t, epsilon_min, epsilon_max)
        
        # failure_types <<- unique(failure_types[failure_types > 0])
        # for (type in .self$failure_types) {
        # cox_model = fit_event_specific_model(t, failure_types, covariates_X, type, sample_ids, t_start, sample_weights, verbose)
        # event_specific_models[[type]] <<- extract_necessary_attributes(cox_model)
        # }
        # },


if __name__ == "__main__":
    pass

