from typing import List, Callable, Optional, Dict, Union
from pandas import Series, DataFrame
import numpy as np
from pymsm.competing_risks_model import CompetingRisksModel
from pymsm.event_specific_fitter import EventSpecificFitter, CoxWrapper
from pymsm.multi_state_competing_risks_model import (
    MultiStateModel,
    default_update_covariates_function,
)


def dict_to_competing_risks_model(model_dict: Dict) -> CompetingRisksModel:
    pass


class MultiStateSimulator(MultiStateModel):

    terminal_states: List[int]
    update_covariates_fn: Callable[[Series, int, int, float, float], Series]
    covariate_names: List[str]
    state_specific_models: Dict[int, CompetingRisksModel]
    covariate_data: DataFrame

    def __init__(
        self,
        terminal_states=[],
        update_covariates_fn=default_update_covariates_function,
        covariate_names=None,
        state_specific_models=dict(),
        covariate_data=None,
    ):
        super().__init__(
            dataset=None,
            terminal_states=terminal_states,
            update_covariates_fn=update_covariates_fn,
            covariate_names=covariate_names,
            event_specific_fitter=CoxWrapper,
            competing_risk_data_format=False,
        )
        self.terminal_states = terminal_states
        self.update_covariates_fn = update_covariates_fn
        self.covariate_names = self._get_covariate_names(covariate_names)
        self.state_specific_models = state_specific_models
        self.covariate_data = None

    def simulate_paths(
        self,
        origin_state: int,
        current_time: int = 0,
        n_random_samples: int = 100,
        max_transitions: int = 10,
    ):
        paths = []
        for idx, covariates in self.covariate_data.iterrows():
            path = self.run_monte_carlo_simulation(
                covariates,
                origin_state,
                current_time,
                n_random_samples,
                max_transitions,
            )
            paths.append(path)
        return paths
