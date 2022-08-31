import pandas as pd
import numpy as np

N_SAMPLES = 3


def test_rotterdam():
    # Load data
    from pymsm.datasets import prep_rotterdam

    dataset, states_labels = prep_rotterdam()
    # Init MultistateModel
    from pymsm.multi_state_competing_risks_model import MultiStateModel

    multi_state_model = MultiStateModel(
        dataset,
        terminal_states=[3],
        state_labels={1: "Primary surgery", 2: "Disease recurrence", 3: "Death"},
    )
    # Plot state diagram
    multi_state_model.plot_state_diagram()

    # Fit to data
    multi_state_model.fit()

    # Run Monte-carlo simulation
    all_mcs = multi_state_model.run_monte_carlo_simulation(
        sample_covariates=dataset[0].covariates.values,
        origin_state=1,
        current_time=0,
        max_transitions=2,
        n_random_samples=N_SAMPLES,
        print_paths=True,
        n_jobs=None,
    )

    assert len(all_mcs) == N_SAMPLES


def test_ebmt():
    # load and prep data
    from pymsm.datasets import prep_ebmt_long

    competing_risk_dataset, covariate_cols, state_labels = prep_ebmt_long()

    # Init
    from pymsm.multi_state_competing_risks_model import (
        MultiStateModel,
        default_update_covariates_function,
    )

    terminal_states = [5, 6]
    multi_state_model = MultiStateModel(
        dataset=competing_risk_dataset,
        terminal_states=terminal_states,
        update_covariates_fn=default_update_covariates_function,
        covariate_names=covariate_cols,
        state_labels=state_labels,
        competing_risk_data_format=True,
    )

    # Fit
    multi_state_model.fit()

    # Run MC for a sample single patient
    mc_paths = multi_state_model.run_monte_carlo_simulation(
        sample_covariates=competing_risk_dataset.loc[0, covariate_cols].values,
        origin_state=1,
        current_time=0,
        n_random_samples=N_SAMPLES,
        max_transitions=10,
        print_paths=False,
    )

    assert len(mc_paths) == N_SAMPLES


def test_covid_hosp():
    # Load and prep data
    from pymsm.datasets import prep_covid_hosp_data, plot_covid_hosp

    dataset, state_labels = prep_covid_hosp_data()
    covariate_cols = ["is_male", "age", "was_severe"]
    covariate_cols = ["is_male", "age", "was_severe"]
    terminal_states = [4]
    state_labels_short = {0: "C", 1: "R", 2: "M", 3: "S", 4: "D"}

    # print single path
    dataset[567].print_path()

    # print path frequencies
    from pymsm.statistics import get_path_frequencies

    path_freqs = get_path_frequencies(dataset, state_labels_short)
    print(path_freqs)
    assert isinstance(path_freqs, pd.Series)

    # define time-varying covariates
    def covid_update_covariates_function(
        covariates_entering_origin_state,
        origin_state=None,
        target_state=None,
        time_at_origin=None,
        abs_time_entry_to_target_state=None,
    ):
        covariates = covariates_entering_origin_state.copy()
        if origin_state == 3:
            covariates["was_severe"] = 1
        return covariates

    # Fit MSM
    from pymsm.multi_state_competing_risks_model import MultiStateModel

    multi_state_model = MultiStateModel(
        dataset=dataset,
        terminal_states=terminal_states,
        update_covariates_fn=covid_update_covariates_function,
        covariate_names=covariate_cols,
        state_labels=state_labels,
    )

    multi_state_model.fit()

    # Run MC for a sample single patient
    mc_paths = multi_state_model.run_monte_carlo_simulation(
        sample_covariates=pd.Series({"is_male": 0, "age": 75, "was_severe": 0}),
        origin_state=2,
        current_time=0,
        n_random_samples=N_SAMPLES,
        max_transitions=10,
        print_paths=False,
        n_jobs=-1,
    )

    # Single patient stats
    from pymsm.statistics import prob_visited_states, stats_total_time_at_states

    # Probability of visiting any of the states
    for state, state_label in state_labels.items():
        if state == 0:
            continue
        print(
            f"Probabilty of ever being {state_label} = {prob_visited_states(mc_paths, states=[state])}"
        )
    # Stats for times at states
    dfs = []
    for state, state_label in state_labels.items():
        if state == 0 or state in terminal_states:
            continue
        dfs.append(
            pd.DataFrame(
                data=stats_total_time_at_states(mc_paths, states=[state]),
                index=[state_label],
            )
        )
    print(pd.concat(dfs).round(3).T)

    path_freqs = get_path_frequencies(mc_paths, state_labels_short)
    print(path_freqs)
    assert isinstance(path_freqs, pd.Series)

    from pymsm.statistics import path_total_time_at_states

    los = np.array(
        [path_total_time_at_states(path, states=[2, 3]) for path in mc_paths]
    )
    print(los)

    assert len(mc_paths) == N_SAMPLES
