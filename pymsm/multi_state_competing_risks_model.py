# -- R source: https://github.com/JonathanSomer/covid-19-multi-state-model/blob/master/model/multi_state_competing_risks_model.R --#

from typing import List, Callable, Optional, Dict
from pandas import Series, DataFrame
import numpy as np
from pymsm.competing_risks_model import CompetingRisksModel


def default_update_covariates_function(covariates_entering_origin_state, origin_state=None, target_state=None,
                                       time_at_origin=None, abs_time_entry_to_target_state=None):
    return covariates_entering_origin_state


RIGHT_CENSORING = 0


class PathObject:
    covariates: Series
    states: List[int]
    time_at_each_state: List[float]
    sample_id: int
    sample_weight: float

    def __init__(self, covariates=None, states=None, time_at_each_state=None, sample_id=None,
                 weight=None):
        """
        PathObject class holds a sample of a single path through the multi state model
        :param covariates: named pandas Series of sample covariates at the initial state
        :param states: list of the states visited (encoded as positive integers, 0 is saved for censoring), in the order
         visited.
        :param time_at_each_state: a list with the duration spent at each state
        Note: if the last state is a terminal state, then the vector of times should be shorter than the vector of
        states by 1. Conversely, if the last state is not a terminal state, then the length of vector times should be
        the same as that of the states.
        :param: optional, int, an identification of this sample
        :param: optional, float, sample weight
        """
        self.covariates = covariates
        self.states = list() if states is None else states
        self.time_at_each_state = list() if time_at_each_state is None else time_at_each_state
        self.sample_id = sample_id
        self.sample_weight = weight
        # This variable is used when simulating paths using monte carlo
        self.stopped_early = None


class MultiStateModel:
    dataset: List[PathObject]
    terminal_states: List[int]
    update_covariates_fn: Callable[[Series, int, int, float, float], Series]
    covariate_names: List[str]
    state_specific_models: Dict[int, CompetingRisksModel]

    def __init__(self, dataset, terminal_states, update_covariates_fn=default_update_covariates_function,
                 covariate_names=None):
        """
        MultiStateModel class fits a competing risks model per state, that is, it treats all state transitions as competing risks.
        See the CompetingRisksModel class.
        :param dataset: list of PathObjects
        :param terminal_states: list of the states which a sample does not leave
        :param update_covariates_fn: update_covariates_function: A state-transition function, which updates the time dependent variables.
        This function is used in fitting the model so that the user doesn't have to manually compute the feautres at
        each state, it is also used in the monte carlo method of predicting the survival curves per sample.
        :param covariate_names: optional list of covariate names to be used in prints
        """
        self.dataset = dataset
        self.terminal_states = terminal_states
        self.update_covariates_fn = update_covariates_fn
        self.covariate_names = self._get_covariate_names(covariate_names)
        self.state_specific_models = dict()
        self._time_is_discrete = None
        self._competing_risk_dataset = None
        self._samples_have_weights = False

        self._assert_valid_input()

    def fit(self, verbose: int = 1) -> None:
        self._competing_risk_dataset = self._prepare_dataset_for_competing_risks_fit()
        self._time_is_discrete = self._check_if_time_is_discrete()

        for state in self._competing_risk_dataset['origin_state'].unique():
            if verbose >= 1:
                print('Fitting Model at State: {}'.format(state))

            model = self._fit_state_specific_model(state, verbose)
            self.state_specific_models[state] = model

    def _assert_valid_input(self) -> None:
        # Check the number os time is either equal or one less than the number of states
        for obj in self.dataset:
            n_states = len(obj.states)
            n_times = len(obj.time_at_each_state)
            assert(n_states == n_times or n_states == n_times+1)

            if n_states == 1 and obj.states[0] in self.terminal_states:
                # TODO - do we want to add printing of the obj ?
                exit("Error: encountered a sample with a single state that is a terminal state.")

        # Check either all objects have an id or none has
        has_id = [obj for obj in self.dataset if obj.sample_id is not None]
        assert(len(has_id) == len(self.dataset) or len(has_id) == 0)

        # Check either all objects have sample weight or none has
        has_weight = [obj for obj in self.dataset if obj.sample_weight is not None]
        assert(len(has_weight) == len(self.dataset) or len(has_weight) == 0)
        self._samples_have_weights = True if len(has_weight) > 0 else False

        # Check all covariates are of the same length
        cov_len = len(self.dataset[0].covariates)
        same_length = [obj for obj in self.dataset if len(obj.covariates) == cov_len]
        assert(len(same_length) == len(self.dataset))

        # Check length of covariate names matches the length of covariates in PathObject
        if self.covariate_names is None:
            return
        assert(len(self.covariate_names) == len(self.dataset[0].covariates))

    def _get_covariate_names(self, covariate_names):
        if covariate_names is not None:
            return covariate_names
        return self.dataset[0].covariates.index.to_list()

    def _check_if_time_is_discrete(self) -> bool:
        times = self._competing_risk_dataset['time_entry_to_origin'].values.tolist() + \
                self._competing_risk_dataset['time_transition_to_target'].values.tolist()
        if all(isinstance(t, int) for t in times):
            return True
        return False

    def _prepare_dataset_for_competing_risks_fit(self) -> DataFrame:
        self._competing_risk_dataset = DataFrame()
        for obj in self.dataset:
            origin_state = obj.states[0]
            covs_entering_origin = Series(dict(zip(self.covariate_names, obj.covariates.values)))
            time_entry_to_origin = 0
            for i, state in enumerate(obj.states):
                transition_row = {}
                time_in_origin = obj.time_at_each_state[i]
                time_transition_to_target = time_entry_to_origin + time_in_origin
                target_state = obj.states[i+1] if i+1 <= len(obj.states) else RIGHT_CENSORING

                # append row corresponding to this transition
                transition_row['sample_id'] = obj.sample_id
                if self._samples_have_weights:
                    transition_row['sample_weight'] = obj.sample_weight
                transition_row['origin_state'] = origin_state
                transition_row['target_state'] = target_state
                transition_row['time_entry_to_origin'] = time_entry_to_origin
                transition_row['time_transition_to_target'] = time_transition_to_target
                transition_row.update(covs_entering_origin.to_dict())
                self._competing_risk_dataset = self._competing_risk_dataset.append(transition_row, ignore_index=True)

                if target_state == RIGHT_CENSORING or target_state in self.terminal_states:
                    break
                else:
                    # Set up for the next iteration
                    covs_entering_origin = self.update_covariates_fn(covs_entering_origin, origin_state, target_state,
                                                                     time_in_origin)
                    origin_state = target_state
                    time_entry_to_origin = time_transition_to_target

        self._competing_risk_dataset['sample_id'] = self._competing_risk_dataset['sample_id'].astype(int)
        self._competing_risk_dataset['origin_state'] = self._competing_risk_dataset['origin_state'].astype(int)
        self._competing_risk_dataset['target_state'] = self._competing_risk_dataset['target_state'].astype(int)
        return self._competing_risk_dataset

    def _fit_state_specific_model(self, state: int, verbose: int = 1):
        state_specific_df = self._competing_risk_dataset[self._competing_risk_dataset['origin_state'] == state]
        state_specific_df.drop(['origin_state'], axis=1, inplace=True)
        state_specific_df.reset_index(drop=True, inplace=True)
        crm = CompetingRisksModel()
        crm.fit(state_specific_df, event_col='target_state', duration_col='time_transition_to_target',
                cluster_col='sample_id', entry_col='time_entry_to_origin', verbose=verbose)
        return crm

    def _run_monte_carlo_simulation(self, sample_covariates, origin_state: int, current_time: int = 0,
                                    n_random_samples: int = 100, max_transitions: int = 10) -> List[PathObject]:
        runs = list()
        for i in range(0, n_random_samples):
            runs.append(self._one_monte_carlo_run(sample_covariates, origin_state, max_transitions, current_time))
        return runs

    def _one_monte_carlo_run(self, sample_covariates, origin_state: int, max_transitions: int,
                             current_time: int = 0) -> PathObject:
        run = PathObject(states=list(), time_at_each_state=list())
        run.stopped_early = False

        current_state = origin_state
        for i in range(0, max_transitions):
            next_state = self._sample_next_state(current_state, sample_covariates, current_time)
            if next_state is None:
                run.stopped_early = True

            time_to_next_state = self._sample_time_to_next_state(current_state, next_state, sample_covariates,
                                                                 current_time)
            run.states.append(current_state)
            run.time_at_each_state.append(time_to_next_state)

            if next_state in self.terminal_states:
                run.states.append(next_state)
                break
            else:
                time_entry_to_target = current_state+time_to_next_state
                sample_covariates = self.update_covariates_fn(sample_covariates, current_state, next_state,
                                                              time_to_next_state, time_entry_to_target)
            current_state = next_state
            current_time = current_time + time_to_next_state

        return run

    def _probability_for_next_state(self, next_state: int, competing_risks_model: CompetingRisksModel, sample_covariates,
                                    t_entry_to_current_state: int = 0):
        unique_event_times = competing_risks_model.unique_event_times(next_state)
        if self._time_is_discrete:
            mask = (unique_event_times > np.floor(t_entry_to_current_state+1))
        else:
            mask = (unique_event_times > t_entry_to_current_state)

        # hazard for the failure type corresponding to 'state':
        hazard = competing_risks_model.hazard_at_unique_event_times(sample_covariates, next_state)
        hazard = hazard[mask]

        # overall survival function evaluated at time of failures corresponding to 'state'
        survival = competing_risks_model.survival_function(unique_event_times[mask], sample_covariates)

        probability_for_state = (hazard*survival).sum()
        return probability_for_state

    def _sample_next_state(self, current_state: int, sample_covariates, t_entry_to_current_state: int) -> Optional[int]:
        competing_risk_model = self.state_specific_models[current_state]
        possible_next_states = competing_risk_model.failure_types

        # compute probabilities for multinomial distribution
        probabilites = {}
        for state in possible_next_states:
            probabilites[state] = self._probability_for_next_state(state, competing_risk_model, sample_covariates,
                                                                   t_entry_to_current_state)

        # when no transition after t_entry_to_current_state was seen
        if all(value == 0 for value in probabilites.values()):
            return None

        mult = np.random.multinomial(1, list(probabilites.values()))
        next_state = possible_next_states[mult.argmax()]
        return next_state

    def _sample_time_to_next_state(self, current_state: int, next_state: int, sample_covariates,
                                   t_entry_to_current_state: int) -> float:
        competing_risk_model = self.state_specific_models[current_state]
        unique_event_times = competing_risk_model.unique_event_times(next_state)

        # ensure discrete variables are sampled from the next time unit
        if self._time_is_discrete:
            mask = (unique_event_times > np.floor(t_entry_to_current_state+1))
        else:
            mask = (unique_event_times > t_entry_to_current_state)
        unique_event_times = unique_event_times[mask]

        # hazard for the failure type corresponding to 'state'
        hazard = competing_risk_model.hazard_at_unique_event_times(sample_covariates, next_state)
        hazard = hazard[mask]

        # overall survival function evaluated at time of failures corresponding to 'state'
        survival = competing_risk_model.survival_function(unique_event_times, sample_covariates)

        probability_for_each_t = (hazard*survival).cumsum()
        probability_for_each_t_given_next_state = probability_for_each_t/probability_for_each_t.max()

        # take the first event time whose probability is less than or equal to eps
        # if we drew a very small eps, use the minimum observed time
        eps = np.random.uniform(size=1)
        possible_times = np.concatenate((unique_event_times[probability_for_each_t_given_next_state <= eps],
                                         [unique_event_times[0]]))
        time_to_next_state = possible_times.max()
        time_to_next_state = time_to_next_state - t_entry_to_current_state

        return time_to_next_state
