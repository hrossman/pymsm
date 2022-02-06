from typing import List, Callable, Optional, Dict, Union
from pandas import Series, DataFrame
import numpy as np
from pymsm.competing_risks_model import CompetingRisksModel, EventSpecificModel
from pymsm.event_specific_fitter import ManualCoxWrapper, CoxWrapper
from pymsm.multi_state_competing_risks_model import (
    MultiStateModel,
    default_update_covariates_function,
)


class MultiStateSimulator(MultiStateModel):

    competing_risks_models_list: List[Dict]
    terminal_states: List[int]
    update_covariates_fn: Callable[[Series, int, int, float, float], Series]
    covariate_names: List[str]

    def __init__(
        self,
        competing_risks_models_list,
        terminal_states,
        update_covariates_fn=default_update_covariates_function,
        covariate_names=None,
    ):

        # Configure the MSM
        super().__init__(
            dataset=None,
            terminal_states=terminal_states,
            update_covariates_fn=update_covariates_fn,
            covariate_names=covariate_names,
            event_specific_fitter=ManualCoxWrapper,
            competing_risk_data_format=True,
        )

        # Configure each competing risks model
        for competing_risks_model_dict in competing_risks_models_list:
            self._configure_competing_risks_model(competing_risks_model_dict)

    def _configure_competing_risks_model(self, competing_risks_model_dict):
        origin_state = competing_risks_model_dict["origin_state"]
        crm = CompetingRisksModel(event_specific_fitter=ManualCoxWrapper)
        self.state_specific_models[origin_state] = crm
        self.state_specific_models[origin_state].failure_types = []
        for i, failure_type in enumerate(competing_risks_model_dict["target_states"]):
            coefs, baseline_hazard = (
                competing_risks_model_dict["model_defs"]["coefs"],
                competing_risks_model_dict["model_defs"]["baseline_hazard"],
            )
            crm.event_specific_models = {
                failure_type: EventSpecificModel(
                    failure_type=failure_type,
                    model=ManualCoxWrapper(coefs, baseline_hazard),
                )
            }

            self.state_specific_models[origin_state].event_specific_models[
                failure_type
            ].extract_necessary_attributes()
            self.state_specific_models[origin_state].failure_types.append(failure_type)

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


def test_on_rossi():
    # Load rossi dataset
    from lifelines.datasets import load_rossi

    rossi = load_rossi()
    from lifelines import CoxPHFitter

    cph = CoxPHFitter()
    cph.fit(rossi, duration_col="week", event_col="arrest")
    baseline_hazard = cph.baseline_hazard_["baseline hazard"]
    coefs = cph.params_

    from pymsm.datasets import load_rossi_competing_risk_data

    rossi_competing_risk_data, covariate_names = load_rossi_competing_risk_data()

    # Define the full model
    competing_risks_models_list = [
        {
            "origin_state": 1,
            "target_states": [2],
            "model_defs": {"coefs": coefs, "baseline_hazard": baseline_hazard},
        }
    ]

    # Configure the simulator
    mssim = MultiStateSimulator(
        competing_risks_models_list,
        terminal_states=[2],
        update_covariates_fn=default_update_covariates_function,
        covariate_names=covariate_names,
    )

    # Run simulation
    mc_paths = mssim.run_monte_carlo_simulation(
        sample_covariates=rossi_competing_risk_data.loc[0, covariate_names],
        origin_state=1,
        current_time=0,
        n_random_samples=5,
        max_transitions=10,
    )
    # Print paths
    for mc_path in mc_paths:
        states = mc_path.states
        time_at_each_state = mc_path.time_at_each_state
        print(states)
        print(time_at_each_state)


if __name__ == "__main__":
    test_on_rossi()
    print("done")

