from typing import List, Callable, Optional, Dict, Union
from pandas import Series, DataFrame
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from pymsm.competing_risks_model import CompetingRisksModel
from pymsm.event_specific_fitter import CoxWrapper, EventSpecificFitter

np.seterr(divide="ignore", invalid="ignore")  # TODO


def default_update_covariates_function(
    covariates_entering_origin_state,
    origin_state=None,
    target_state=None,
    time_at_origin=None,
    abs_time_entry_to_target_state=None,
):
    return covariates_entering_origin_state


RIGHT_CENSORING = 0


class PathObject:
    """This class holds all necessary attributes of a single path for the multi state model.

    Args:
        covariates (Series, optional): Named pandas Series of sample covariates at the initial state. Defaults to None.
        states (List[int], optional): States visited (encoded as positive integers, 0 is saved for censoring), in the order visited. Defaults to None.
        time_at_each_state (List[float], optional): Time at each state. Defaults to None.
        sample_id (int, optional): An identification of this sample. Defaults to None.
        weight (float, optional): Sample weight. Defaults to None.

    Note:
        If the last state is a terminal state, then the vector of times should be shorter than the vector of
        states by 1. Conversely, if the last state is not a terminal state, then the length of vector times should be
        the same as that of the states.

    """

    def __init__(
        self,
        covariates: Series = None,
        states: List[int] = None,
        time_at_each_state: List[float] = None,
        sample_id: int = None,
        weight: float = None,
    ):

        self.covariates = covariates
        self.states = list() if states is None else states
        self.time_at_each_state = (
            list() if time_at_each_state is None else time_at_each_state
        )
        self.sample_id = sample_id
        self.sample_weight = weight
        # This variable is used when simulating paths using monte carlo
        self.stopped_early = None

    def print_path(self):
        """Helper function for printing the paths of a Monte Carlo simulation"""
        if self.sample_id is not None:
            print(f"Sample id: {self.sample_id}")
        print(f"States: {self.states}")
        print(f"Transition times: {self.time_at_each_state}")
        if self.covariates is not None:
            print(f"Covariates:\n{self.covariates}")


class MultiStateModel:
    """This class fits a competing risks model per state, that is, it treats all state transitions as competing risks. See the CompetingRisksModel class

    Args:
        dataset (Union[List[PathObject], DataFrame]): either a list of PathObject or a pandas data frame in the format used to fit the CompetingRiskModel class. Dataset used to fit a competing risk model to each state
        terminal_states (List[int]): States which a sample does not leave
        update_covariates_fn (Callable[ [Series, int, int, float, float], Series ], optional): A state-transition function, which updates the time dependent variables. This function is used in fitting the model so that the user doesn't have to manually compute the features at each state, it is also used in the monte carlo method of predicting the survival curves per sample. Defaults to default_update_covariates_function.
        covariate_names (List[str], optional): Optional list of covariate names to be used. Defaults to None.
        event_specific_fitter (EventSpecificFitter, optional): This class holds the model that will be fitter inside the CompetingRisksModel. Defaults to CoxWrapper.
        competing_risk_data_format (bool, optional): A boolean indicating the format of the dataset parmeter, if False - the dataset is assumed to be a list of PathObjects, if True - the dataset is assumed to be a dataframe which is compatible in format for fitting the CompetingRiskModel class. Defaults to False.

    Attributes:
        state_specific_models (Dict[int, CompetingRisksModel]): A dictionary of CompetingRisksModel objects, one for each state. Available after running the "fit" function.

    Note:
        The update_covariates_fn could be any function you choose to write, but it needs to have the following parameter
        types (in this order): pandas Series, int, int, float, float; and return a pandas Series.
    """

    def __init__(
        self,
        dataset: Union[List[PathObject], DataFrame],
        terminal_states: List[int],
        update_covariates_fn: Callable[
            [Series, int, int, float, float], Series
        ] = default_update_covariates_function,
        covariate_names: List[str] = None,
        event_specific_fitter: EventSpecificFitter = CoxWrapper,
        competing_risk_data_format: bool = False,
    ):
        self.dataset = dataset
        self.terminal_states = terminal_states
        self.update_covariates_fn = update_covariates_fn
        self.covariate_names = self._get_covariate_names(covariate_names)
        self.state_specific_models: Dict[int, CompetingRisksModel] = dict()
        self._time_is_discrete: bool = None
        self.competing_risk_dataset: DataFrame = None
        self._samples_have_weights: bool = False
        self._competing_risk_data_format = competing_risk_data_format
        self._event_specific_fitter = event_specific_fitter

        if not self._competing_risk_data_format:
            self._assert_valid_input()

    def fit(self, verbose: int = 1) -> None:
        """Fit a CompetingRiskModel for each state

        Args:
            verbose (int, optional): verbosity, by default 1. Defaults to 1.
        """

        self.competing_risk_dataset = (
            self.dataset
            if self._competing_risk_data_format
            else self._prepare_dataset_for_competing_risks_fit()
        )
        self._time_is_discrete = self._check_if_time_is_discrete()

        for state in self.competing_risk_dataset["origin_state"].unique():
            if verbose >= 1:
                print("Fitting Model at State: {}".format(state))

            model = self._fit_state_specific_model(state, verbose)
            self.state_specific_models[state] = model

    def _assert_valid_input(self) -> None:
        """Checks that the dataset is valid for running the multi state competing risk model"""
        # Check the number os time is either equal or one less than the number of states
        for obj in self.dataset:
            n_states = len(obj.states)
            n_times = len(obj.time_at_each_state)
            assert n_states == n_times or n_states == n_times + 1

            if n_states == 1 and obj.states[0] in self.terminal_states:
                obj.print_path()
                exit(
                    "Error: encountered a sample with a single state that is a terminal state."
                )

        # Check either all objects have an id or none has
        has_id = [obj for obj in self.dataset if obj.sample_id is not None]
        assert len(has_id) == len(self.dataset) or len(has_id) == 0

        # Check either all objects have sample weight or none has
        has_weight = [obj for obj in self.dataset if obj.sample_weight is not None]
        assert len(has_weight) == len(self.dataset) or len(has_weight) == 0
        self._samples_have_weights = True if len(has_weight) > 0 else False

        # Check all covariates are of the same length
        cov_len = len(self.dataset[0].covariates)
        same_length = [obj for obj in self.dataset if len(obj.covariates) == cov_len]
        assert len(same_length) == len(self.dataset)

        # Check length of covariate names matches the length of covariates in PathObject
        if self.covariate_names is None:
            return
        assert len(self.covariate_names) == len(self.dataset[0].covariates)

    def _get_covariate_names(self, covariate_names: List[str]) -> List[str]:
        """This functions sets the covariate names that will be used in prints.
            Names are taken either from the given covariate names provided by the user,
            or if None provided - from the named pandas Series of covariates of the PathObject in the dataset

        Args:
            covariate_names (List[str], optional): covariate names provided in class init

        Returns:
            List: List of covariate names
        """

        if covariate_names is not None:
            return covariate_names
        return self.dataset[0].covariates.index.to_list()

    def _check_if_time_is_discrete(self) -> bool:
        """This function check whether the time in the dataset is discrete"""
        times = (
            self.competing_risk_dataset["time_entry_to_origin"].values.tolist()
            + self.competing_risk_dataset["time_transition_to_target"].values.tolist()
        )
        if all(isinstance(t, int) for t in times):
            return True
        return False

    def _prepare_dataset_for_competing_risks_fit(self) -> DataFrame:
        """This function converts the given dataset (list of PathObjects) to a pandas DataFrame that will be used when
        fitting the CompetingRiskModel class
        """
        self.competing_risk_dataset = DataFrame()
        for obj in self.dataset:
            origin_state = obj.states[0]
            covs_entering_origin = Series(
                dict(zip(self.covariate_names, obj.covariates.values))
            )
            time_entry_to_origin = 0
            for i, state in enumerate(obj.states):
                transition_row = {}
                time_in_origin = obj.time_at_each_state[i]
                time_transition_to_target = time_entry_to_origin + time_in_origin
                target_state = (
                    obj.states[i + 1] if i + 1 < len(obj.states) else RIGHT_CENSORING
                )

                # append row corresponding to this transition
                transition_row["sample_id"] = obj.sample_id
                if self._samples_have_weights:
                    transition_row["sample_weight"] = obj.sample_weight
                transition_row["origin_state"] = origin_state
                transition_row["target_state"] = target_state
                transition_row["time_entry_to_origin"] = time_entry_to_origin
                transition_row["time_transition_to_target"] = time_transition_to_target
                transition_row.update(covs_entering_origin.to_dict())
                self.competing_risk_dataset = self.competing_risk_dataset.append(
                    transition_row, ignore_index=True
                )
                # TODO change to concat due to:
                # FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.

                if (
                    target_state == RIGHT_CENSORING
                    or target_state in self.terminal_states
                ):
                    break
                else:
                    # Set up for the next iteration
                    covs_entering_origin = self.update_covariates_fn(
                        covs_entering_origin, origin_state, target_state, time_in_origin
                    )
                    origin_state = target_state
                    time_entry_to_origin = time_transition_to_target

        self.competing_risk_dataset["sample_id"] = self.competing_risk_dataset[
            "sample_id"
        ].astype(int)
        self.competing_risk_dataset["origin_state"] = self.competing_risk_dataset[
            "origin_state"
        ].astype(int)
        self.competing_risk_dataset["target_state"] = self.competing_risk_dataset[
            "target_state"
        ].astype(int)
        return self.competing_risk_dataset

    def _fit_state_specific_model(
        self, state: int, verbose: int = 1
    ) -> CompetingRisksModel:
        """Fit a CompetingRiskModel for a specific given state

        Args:
            state (int): State to fit the model for
            verbose (int, optional): verbosity. Defaults to 1.

        Returns:
            CompetingRisksModel: state specific model
        """

        state_specific_df = self.competing_risk_dataset[
            self.competing_risk_dataset["origin_state"] == state
        ].copy()
        state_specific_df.drop(["origin_state"], axis=1, inplace=True)
        state_specific_df.reset_index(drop=True, inplace=True)
        crm = CompetingRisksModel(self._event_specific_fitter)
        crm.fit(
            state_specific_df,
            event_col="target_state",
            duration_col="time_transition_to_target",
            cluster_col="sample_id",
            entry_col="time_entry_to_origin",
            verbose=verbose,
        )
        return crm

    def _assert_valid_simulation_input(
        self,
        sample_covariates: np.ndarray,
        origin_state: int,
        current_time: int,
        n_random_samples: int,
        max_transitions: int,
        n_jobs: int,
        print_paths: bool,
    ):
        """This function checks if the input to the simulation is valid."""

        # TODO assert valid inputs for sample_covariates (Series or np.ndarray, and length), origin_state
        assert current_time >= 0
        assert isinstance(n_random_samples, int)
        assert n_random_samples > 0
        assert isinstance(max_transitions, int)
        assert max_transitions > 0
        assert isinstance(n_jobs, int)
        assert n_jobs >= -1
        assert isinstance(print_paths, bool)

    def run_monte_carlo_simulation(
        self,
        sample_covariates: np.ndarray,  # TODO change to np.ndarray OR pd.Series
        origin_state: int,
        current_time: int = 0,
        n_random_samples: int = 100,
        max_transitions: int = 10,
        n_jobs: int = -1,
        print_paths: bool = False,
    ) -> List[PathObject]:
        """This function samples random paths using Monte Carlo simulation.
            These paths will be used for prediction for a single sample.
            Initial sample covariates, along with the sample’s current state are supplied.
            The next states are sequentially sampled via the model parameters.
            The process concludes when the sample arrives at a terminal state or the number of transitions exceeds the
            specified maximum.

        Args:
            sample_covariates (np.ndarray): Initial sample covariates, when entering the origin state
            origin_state (int): Initial state where the path begins from
            current_time (int, optional): Time when starting the sample path. Defaults to 0.
            n_random_samples (int, optional): Number of random paths to create. Defaults to 100.
            max_transitions (int, optional): Max number of transitions to allow in the paths. Defaults to 10.
            n_jobs (int, optional): Number of parallel jobs to run. Defaults to -1.
            print_paths (bool, optional): Whether to print the paths or not. Defaults to False.

        Returns:
            List[PathObject]: list of length n_random_samples, contining the randomly create PathObjects
        """
        # Check input is valid
        self._assert_valid_simulation_input(
            sample_covariates,
            origin_state,
            current_time,
            n_random_samples,
            max_transitions,
            n_jobs,
            print_paths,
        )

        if n_jobs is None:  # no parallelization
            runs = []
            for i in tqdm(range(0, n_random_samples)):
                runs.append(
                    self._one_monte_carlo_run(
                        sample_covariates, origin_state, max_transitions, current_time
                    )
                )
        else:  # Run parallel jobs
            runs = Parallel(n_jobs=n_jobs)(
                delayed(self._one_monte_carlo_run)(
                    sample_covariates, origin_state, max_transitions, current_time
                )
                for i in tqdm(range(0, n_random_samples))
            )

        if print_paths:
            self._print_paths(runs)
        return runs

    def _one_monte_carlo_run(
        self,
        sample_covariates: np.ndarray,
        origin_state: int,
        max_transitions: int,
        current_time: int = 0,
    ) -> PathObject:
        """This function create one path using Monte Carlo Simulations.
        See documentation of run_monte_carlo_simulation.
        """
        run = PathObject(states=list(), time_at_each_state=list())
        run.stopped_early = False

        current_state = origin_state
        for i in range(0, max_transitions):
            next_state = self._sample_next_state(
                current_state, sample_covariates, current_time
            )
            if next_state is None:
                run.stopped_early = True
                return run

            time_to_next_state = self._sample_time_to_next_state(
                current_state, next_state, sample_covariates, current_time
            )
            run.states.append(current_state)
            run.time_at_each_state.append(time_to_next_state)

            if next_state in self.terminal_states:
                run.states.append(next_state)
                break
            else:
                time_entry_to_target = current_state + time_to_next_state
                sample_covariates = self.update_covariates_fn(
                    sample_covariates,
                    current_state,
                    next_state,
                    time_to_next_state,
                    time_entry_to_target,
                )
            current_state = next_state
            current_time = current_time + time_to_next_state

        return run

    def _probability_for_next_state(
        self,
        next_state: int,
        competing_risks_model: CompetingRisksModel,
        sample_covariates: np.ndarray,
        t_entry_to_current_state: int = 0,
    ):
        """This function calculates the probability of transition to the next state, using the competing_risks_model
        model parameters
        """
        unique_event_times = competing_risks_model.unique_event_times(next_state)
        if self._time_is_discrete:
            mask = unique_event_times > np.floor(t_entry_to_current_state + 1)
        else:
            mask = unique_event_times > t_entry_to_current_state

        # hazard for the failure type corresponding to 'state':
        hazard = competing_risks_model.hazard_at_unique_event_times(
            sample_covariates, next_state
        )
        hazard = hazard[mask]

        # overall survival function evaluated at time of failures corresponding to 'state'
        survival = competing_risks_model.survival_function(
            unique_event_times[mask], sample_covariates
        )

        probability_for_state = np.nansum(hazard * survival)
        return probability_for_state

    def _sample_next_state(
        self,
        current_state: int,
        sample_covariates: np.ndarray,
        t_entry_to_current_state: int,
    ) -> Optional[int]:
        """This function samples the next state, according to a multinomial distribution, using probabilites defines
        by _probability_for_next_state function.
        """
        competing_risk_model = self.state_specific_models[current_state]
        possible_next_states = competing_risk_model.failure_types

        # compute probabilities for multinomial distribution
        probabilites = {}
        for state in possible_next_states:
            probabilites[state] = self._probability_for_next_state(
                state, competing_risk_model, sample_covariates, t_entry_to_current_state
            )

        # when no transition after t_entry_to_current_state was seen
        if all(value == 0 for value in probabilites.values()):
            return None

        mult = np.random.multinomial(1, list(probabilites.values()))
        next_state = possible_next_states[mult.argmax()]
        return next_state

    def _sample_time_to_next_state(
        self,
        current_state: int,
        next_state: int,
        sample_covariates: np.ndarray,
        t_entry_to_current_state: int,
    ) -> float:
        """This function samples the time of transition to the next state, using the hazard and survival provided by
        the competing risk model of the current_state
        """
        competing_risk_model = self.state_specific_models[current_state]
        unique_event_times = competing_risk_model.unique_event_times(next_state)

        # ensure discrete variables are sampled from the next time unit
        if self._time_is_discrete:
            mask = unique_event_times > np.floor(t_entry_to_current_state + 1)
        else:
            mask = unique_event_times > t_entry_to_current_state
        unique_event_times = unique_event_times[mask]

        # hazard for the failure type corresponding to 'state'
        hazard = competing_risk_model.hazard_at_unique_event_times(
            sample_covariates, next_state
        )
        hazard = hazard[mask]

        # overall survival function evaluated at time of failures corresponding to 'state'
        survival = competing_risk_model.survival_function(
            unique_event_times, sample_covariates
        )

        probability_for_each_t = np.nancumsum(hazard * survival)
        probability_for_each_t_given_next_state = (
            probability_for_each_t / probability_for_each_t.max()
        )  # TODO this raises warnings and we should create better error handling

        # take the first event time whose probability is less than or equal to eps
        # if we drew a very small eps, use the minimum observed time
        eps = np.random.uniform(size=1)
        possible_times = np.concatenate(
            (
                unique_event_times[probability_for_each_t_given_next_state <= eps],
                [unique_event_times[0]],
            )
        )
        time_to_next_state = possible_times.max()
        time_to_next_state = time_to_next_state - t_entry_to_current_state

        return time_to_next_state

    def _print_paths(self, mc_paths):
        """Helper function for printing the paths of a Monte Carlo simulation"""
        for mc_path in mc_paths:
            mc_path.print_path()
            print("\n")
