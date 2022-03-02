from typing import List, Callable, Optional, Dict, Union, Tuple
from pandas import Series, DataFrame
import numpy as np
from pymsm.competing_risks_model import CompetingRisksModel, EventSpecificModel
from pymsm.event_specific_fitter import ManualCoxWrapper, CoxWrapper
from pymsm.multi_state_competing_risks_model import (
    MultiStateModel,
    default_update_covariates_function,
)


def _extract_model_parts(
    event_specific_model: EventSpecificModel,
) -> Tuple[Series, Series]:
    """Helper function for extracting (coefs, baseline_hazard) from an event specific model"""
    coefs = event_specific_model.model._model.params_
    baseline_hazard = event_specific_model.model._model.baseline_hazard_[
        "baseline hazard"
    ]
    return coefs, baseline_hazard


def extract_competing_risks_models_list_from_msm(
    multi_state_model: MultiStateModel, verbose: bool = False
) -> List[Dict]:
    """Extract competing risks models list from a MultiStateModel"""
    competing_risks_models_list = []
    for origin_state, crm in multi_state_model.state_specific_models.items():
        state_dict = {
            "origin_state": origin_state,
            "target_states": [],
            "model_defs": [],
        }
        for target_state, event_specific_model in crm.event_specific_models.items():
            if verbose:
                print(origin_state, target_state, event_specific_model)
            coefs, baseline_hazard = _extract_model_parts(event_specific_model)
            state_dict["target_states"].append(target_state)
            state_dict["model_defs"].append(
                {"coefs": coefs, "baseline_hazard": baseline_hazard}
            )
        competing_risks_models_list.append(state_dict)
    return competing_risks_models_list


class MultiStateSimulator(MultiStateModel):
    """This class configures a multi-state model simulator from predefined model parts. It inherents from the MultiStateModel class, but instead of being fitted to data, it is configured based on predefined models.

    Args:
        competing_risks_models_list (List[Dict]): A list of dictionaries, each of which contains the following keys:
            origin_state: The state from which the simulation starts.
            target_states: A list of states to which the simulation transitions.
            model_defs: A dictionary containing the following keys:
                coefs: A list of coefficients for the competing risks model.
                baseline_hazard: A list of baseline hazards for the competing risks model.
        terminal_states (List[int]): A list of states that are terminal states.
        update_covariates_fn (Callable[ [Series, int, int, float, float], Series ], optional): A function that takes in a covariate dataframe and returns a covariate dataframe.. Defaults to default_update_covariates_function.
        covariate_names (List[str], optional): A list of covariate names.. Defaults to None.

    Note:
        The update_covariates_fn could be any function you choose to write, but it needs to have the following parameter
        types (in this order): pandas Series, int, int, float, float,
        and return a pandas Series.

    """

    def __init__(
        self,
        competing_risks_models_list: List[Dict],
        terminal_states: List[int],
        update_covariates_fn: Callable[
            [Series, int, int, float, float], Series
        ] = default_update_covariates_function,
        covariate_names: List[str] = None,
    ):
        # Configure the MSM, dataset is not relevant
        super().__init__(
            dataset=None,
            terminal_states=terminal_states,
            update_covariates_fn=update_covariates_fn,
            covariate_names=covariate_names,
            event_specific_fitter=ManualCoxWrapper,
            competing_risk_data_format=True,
        )

        # Iterate and configure each competing risks model
        for competing_risks_model_dict in competing_risks_models_list:
            self._configure_competing_risks_model(competing_risks_model_dict)

    def _configure_competing_risks_model(self, competing_risks_model_dict):
        """Helper function to configure a competing risks model from a dictionary containing the following keys:
        origin_state: The state from which the simulation starts.
        target_states: A list of states to which the simulation transitions.
        model_defs: A dictionary containing the following keys:
            coefs: A list of coefficients for the competing risks model.
            baseline_hazard: A list of baseline hazards for the competing risks model.
        """
        origin_state = competing_risks_model_dict["origin_state"]

        # Init CompetingRisksModel
        crm = CompetingRisksModel(event_specific_fitter=ManualCoxWrapper)
        crm.event_specific_models = {}
        crm.failure_types = []
        self.state_specific_models[origin_state] = crm

        # Iterate and configure an EventSpecificModel for each origin_state
        for i, failure_type in enumerate(competing_risks_model_dict["target_states"]):
            coefs, baseline_hazard = (
                competing_risks_model_dict["model_defs"][i]["coefs"],
                competing_risks_model_dict["model_defs"][i]["baseline_hazard"],
            )
            self.state_specific_models[origin_state].event_specific_models[
                failure_type
            ] = EventSpecificModel(
                failure_type=failure_type,
                model=ManualCoxWrapper(coefs, baseline_hazard),
            )
            self.state_specific_models[origin_state].event_specific_models[
                failure_type
            ].extract_necessary_attributes()
            self.state_specific_models[origin_state].failure_types.append(failure_type)


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
            "model_defs": [{"coefs": coefs, "baseline_hazard": baseline_hazard}],
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
        print_paths=True,
    )


if __name__ == "__main__":
    test_on_rossi()
    print("done")
