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
    def __init__(
        self,
        multistate_model_dict: Dict = {
            "terminal_states": [],
            "update_covariates_fn": default_update_covariates_function,
            "covariate_names": [],
            "competing_risks_models_list": [
                {
                    "origin_state": None,
                    "target_states": [],
                    "model_defs": {"coefs": None, "baseline_hazard": None},
                }
            ],
        },
        covariate_data: DataFrame = None,
    ):

        # Configure the MSM
        super().__init__(
            dataset=None,
            terminal_states=multistate_model_dict["terminal_states"],
            update_covariates_fn=multistate_model_dict["update_covariates_fn"],
            covariate_names=multistate_model_dict["covariate_names"],
            event_specific_fitter=ManualCoxWrapper,
            competing_risk_data_format=True,
        )

        # Configure each competing risks model
        for competing_risks_model_dict in multistate_model_dict[
            "competing_risks_models_list"
        ]:
            self._configure_competing_risks_model(competing_risks_model_dict)

        self.covariate_data = None

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
    # Load roossi dataset
    from pymsm.datasets import load_rossi_competing_risk_data

    rossi_competing_risk_data, covariate_names = load_rossi_competing_risk_data()

    # Define the full model
    multistate_model_dict = {
        "terminal_states": [2],
        "update_covariates_fn": default_update_covariates_function,
        "covariate_names": ["fin", "age", "race", "wexp", "mar", "paro", "prio"],
        "competing_risks_models_list": [
            {
                "origin_state": 1,
                "target_states": [2],
                "model_defs": {"coefs": coefs, "baseline_hazard": baseline_hazard},
            }
        ],
    }


if __name__ == "__main__":
    test_on_rossi()
