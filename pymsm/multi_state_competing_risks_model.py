# -- R source: https://github.com/JonathanSomer/covid-19-multi-state-model/blob/master/model/multi_state_competing_risks_model.R --#

from typing import List, Callable
from pandas import Series, DataFrame


def default_update_covariates_function(covariates_entering_origin_state, origin_state=None, target_state=None,
                                       time_at_origin=None):
    return covariates_entering_origin_state


RIGHT_CENSORING = 0


class PathObject:
    def __init__(self, covariates: Series, states: List, time_at_each_state: List, sample_id: int = None):
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

        """
        self.covariates = covariates
        self.states = states
        self.time_at_each_state = time_at_each_state
        self.sample_id = sample_id


class MultiStateModel:
    def __init__(self, dataset: List[PathObject], terminal_states: List,
                 update_covariates_fn: Callable = default_update_covariates_function,
                 covariate_names: List[str] = None):
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
        self.state_specific_models = list()
        self._time_is_discrete = None
        self._competing_risk_dataset = None

        self._assert_valid_input()

    def fit(self, verbose: int = 1) -> None:
        self._competing_risk_dataset = self._prepare_dataset_for_competing_risks_fit()
        self._time_is_discrete = self._check_if_time_is_discrete()

        for state in self._competing_risk_dataset['origin_state'].unique():
            if verbose >= 1:
                print('Fitting Model at State: {}'.format(state))

            model = self._fit_state_specific_model(state, self._competing_risk_dataset, verbose)
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

        # Check either all objects have and id or none has
        has_id = [obj for obj in self.dataset if obj.sample_id is not None]
        assert(len(has_id) == len(self.dataset) or len(has_id) == 0)

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
        self._competing_risk_dataset = DataFrame(columns=['sample_id', 'origin_state', 'target_state',
                                                          'time_entry_to_origin', 'time_transition_to_target'] +
                                                 self.covariate_names)
        for obj in self.dataset:
            origin_state = obj.states[0]
            covs_entering_origin = obj.covariates
            time_entry_to_origin = 0
            for i, state in enumerate(obj.states):
                transition_row = {}
                time_in_origin = obj.time_at_each_state[i]
                time_transition_to_target = time_entry_to_origin + time_in_origin
                target_state = obj.states[i+1] if i+1 <= len(obj.states) else RIGHT_CENSORING

                # append row corresponding to this transition
                transition_row['sample_id'] = obj.sample_id
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

        return self._competing_risk_dataset

    def _fit_state_specific_model(self, state: int, competing_risk_dataset: DataFrame, verbose: int = 1):
        pass

    def _run_monte_carlo_simulation(self, sample_covariates, origin_state: int, current_time: int = 0,
                                    n_random_sampels: int =  100, max_transitions: int = 10):
        pass

    def _one_monte_carlo_run(self, sample_covariates, origin_state: int, max_transitions: int, current_time: int = 0):
        pass

    def _probability_for_next_state(self, next_state: int, competing_risks_model, sample_covariates,
                                    t_entry_to_current_state : int = 0):
        pass

    def _sample_next_state(self, current_state: int, sample_covariates, t_entry_to_current_state: int):
        pass

    def _sample_time_to_next_state(self, current_state: int, next_state: int, sample_covariates,
                                   t_entry_to_current_state: int):
        pass