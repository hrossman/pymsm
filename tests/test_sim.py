N_SAMPLES = 5


def test_msm_sim():
    # Load Rotterdam data and fit a model
    from pymsm.datasets import prep_rotterdam

    dataset, state_labels = prep_rotterdam()
    terminal_states = [3]
    from pymsm.multi_state_competing_risks_model import (
        MultiStateModel,
        default_update_covariates_function,
    )

    multi_state_model = MultiStateModel(
        dataset, terminal_states, default_update_covariates_function
    )
    multi_state_model.fit()

    # Extract the model parameters
    from pymsm.simulation import extract_competing_risks_models_list_from_msm

    competing_risks_models_list = extract_competing_risks_models_list_from_msm(
        multi_state_model, verbose=True
    )

    from pymsm.simulation import MultiStateSimulator

    # Configure the simulator
    mssim = MultiStateSimulator(
        competing_risks_models_list,
        terminal_states=terminal_states,
        update_covariates_fn=default_update_covariates_function,
        covariate_names=[
            "year",
            "age",
            "meno",
            "grade",
            "nodes",
            "pgr",
            "er",
            "hormon",
            "chemo",
        ],
    )

    # Run MC for a sample single patient
    sim_paths = mssim.run_monte_carlo_simulation(
        sample_covariates=dataset[0].covariates.values,
        origin_state=1,
        current_time=2,
        n_random_samples=N_SAMPLES,
        max_transitions=10,
        print_paths=True,
        n_jobs=1,
    )

    assert len(sim_paths) == N_SAMPLES


def test_sim_on_rossi():
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

    from pymsm.multi_state_competing_risks_model import (
        default_update_covariates_function,
    )
    from pymsm.simulation import MultiStateSimulator

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

    assert len(mc_paths) == N_SAMPLES
