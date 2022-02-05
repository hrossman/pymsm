from typing import List, Callable, Optional, Dict, Union
from pandas import Series, DataFrame
import numpy as np
from pymsm.competing_risks_model import CompetingRisksModel
from pymsm.event_specific_fitter import EventSpecificFitter, CoxWrapper
from pymsm.multi_state_competing_risks_model import (
    MultiStateCompetingRisksModel,
    default_update_covariates_function,
)


class MultiStateSimulator(MultiStateCompetingRisksModel):

    terminal_states: List[int]
    update_covariates_fn: Callable[[Series, int, int, float, float], Series]
    covariate_names: List[str]
    state_specific_models: Dict[int, CompetingRisksModel]

    def __init__(
        self,
        terminal_states,
        update_covariates_fn=default_update_covariates_function,
        covariate_names=None,
        state_specific_models=dict(),
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

    def simulate_paths():
        pass