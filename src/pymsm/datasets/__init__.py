from re import S
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymsm.multi_state_competing_risks_model import PathObject
from pymsm.utils import get_categorical_columns
from pkg_resources import resource_filename
from lifelines.datasets import load_rossi


def load_rossi_competing_risk_data():
    rossi = load_rossi()
    covariate_names = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
    rossi_competing_risk_data = rossi[covariate_names].copy()
    rossi_competing_risk_data["sample_id"] = rossi_competing_risk_data.index.values
    rossi_competing_risk_data["origin_state"] = 1
    rossi_competing_risk_data["target_state"] = rossi["arrest"].replace({1: 2})
    rossi_competing_risk_data["time_entry_to_origin"] = 0
    rossi_competing_risk_data["time_transition_to_target"] = rossi["week"]
    return rossi_competing_risk_data, covariate_names


def _load_dataset(filename, **kwargs):
    """
    Load a dataset from pymsm.datasets
    Parameters
    ----------
    filename : string
        for example "larynx.csv"
    usecols : list
        list of columns in file to use
    Returns
    -------
        output: DataFrame
    """
    return pd.read_csv(
        resource_filename("pymsm", "datasets/" + filename),
        engine="python",
        index_col=0,
        **kwargs
    )


def load_ebmt(data_format: str = None, **kwargs) -> pd.DataFrame:
    """Load EBMT dataset (from R mstate package). We consider survival after a transplant treatment of patients suffering from a blood cancer. The data have been provided by the EBMT (the European Group for Blood and Marrow Transplantation); they are available in mstate as ebmt4. The present data set has been compiled to illustrate the models and the software. To facilitate this illustration, only patients with complete covariate information and a reasonable amount of information about intermediate events have been included

    Parameters
    ----------
    data_format : str, optional
        options are "wide", "long"., by default None

    Returns
    -------
    pd.DataFrame
        [description]
    """
    if data_format == "wide":
        data = _load_dataset("ebmt_wide.csv", **kwargs)
    elif data_format == "long":
        data = _load_dataset("ebmt_long.csv", **kwargs)
    else:
        data = _load_dataset("ebmt_long.csv", **kwargs)

    return data


def load_rotterdam() -> pd.DataFrame:
    """Load rotterdam dataset (from R survival package). The rotterdam data set includes 2982 primary breast cancers patients whose data whose records were included in the Rotterdam tumor bank.

    Returns
    -------
    pd.DataFrame
    """
    data = _load_dataset("rotterdam.csv")
    return data.reset_index()


def load_aidssi(**kwargs) -> pd.DataFrame:
    """Load AIDSSI dataset (from R mstate package)

    Parameters
    ----------
    data_format : str, optional
        options are "wide", "long"., by default None

    Returns
    -------
    pd.DataFrame
        [description]
    """
    data = _load_dataset("aidssi.csv", **kwargs)

    return data


def prep_ebmt_long():
    longdata = load_ebmt()
    longdata.loc[longdata["status"] == 0, "to"] = 0  # take care of right censoring
    longdata = longdata.drop("trans", axis=1).drop_duplicates()
    longdata = longdata.sort_values(["id", "Tstart", "from", "status"]).drop_duplicates(
        ["id", "Tstart", "from"], keep="last"
    )
    longdata = longdata.reset_index(drop=True)

    # Categorical columns
    cat_cols = ["match", "proph", "year", "agecl"]
    cat_df = get_categorical_columns(longdata, cat_cols)
    covariate_cols = cat_df.columns
    data = pd.concat([longdata.drop(cat_cols, axis=1), cat_df], axis=1)

    rename_cols = {
        "id": "sample_id",
        "from": "origin_state",
        "to": "target_state",
        "Tstart": "time_entry_to_origin",
        "Tstop": "time_transition_to_target",
    }

    competing_risk_dataset = data[rename_cols.keys()].rename(columns=rename_cols)
    competing_risk_dataset = pd.concat(
        [competing_risk_dataset, data[covariate_cols]], axis=1
    )

    states_labels = {
        1: "Transplant",
        2: "Rec",
        3: "AE",
        4: "AE & Rec",
        5: "Relapse",
        6: "Death",
    }

    return competing_risk_dataset, covariate_cols, states_labels


def prep_rotterdam():
    rotterdam = load_rotterdam()
    dataset = []
    eps = 0.1
    cov_cols = ["year", "age", "meno", "grade", "nodes", "pgr", "er", "hormon", "chemo"]
    for index, row in rotterdam.iterrows():
        path = PathObject(covariates=row[cov_cols], sample_id=row["pid"])
        if not row["recur"] and not row["death"]:
            path.states = [1]
            path.time_at_each_state = [row["rtime"]]
        if row["recur"] and not row["death"]:
            path.states = [1, 2]
            recur_time = (
                eps if row["dtime"] - row["rtime"] == 0 else row["dtime"] - row["rtime"]
            )
            path.time_at_each_state = [row["rtime"], recur_time]
        if row["death"] and not row["recur"]:
            path.states = [1, 3]
            path.time_at_each_state = [row["dtime"]]
        if row["recur"] and row["death"]:
            path.states = [1, 2, 3]
            death_time = (
                eps if row["dtime"] - row["rtime"] == 0 else row["dtime"] - row["rtime"]
            )
            path.time_at_each_state = [row["rtime"], death_time]
        dataset.append(path)

    states_labels = {1: "Primary surgery", 2: "Disease recurrence", 3: "Death"}
    return dataset, states_labels


def prep_covid_hosp_data():
    """Covid hospitalization data from: https://github.com/JonathanSomer/covid-19-multi-state-model/blob/master/data/data_for_paper.csv"""
    state_cols = [
        "new_type0",
        "new_type1",
        "new_type2",
        "new_type3",
        "new_type4",
        "new_type5",
        "new_type6",
        "new_type7",
        "new_type8",
        "new_type9",
        "new_type10",
    ]
    time_cols = [
        "new_time1",
        "new_time2",
        "new_time3",
        "new_time4",
        "new_time5",
        "new_time6",
        "new_time7",
        "new_time8",
        "new_time9",
        "new_time10",
    ]

    def parse_row(row, verbose=False):
        terminal_states = [4]
        states = row[state_cols].values.astype(int)
        time_at_each_state = row[time_cols].values.astype(float)
        first_nan = np.where(np.isnan(time_at_each_state))[0]  # find first nan
        if len(first_nan) > 0:
            first_nan = first_nan[0]
            states = states[: (first_nan + 1)]
            time_at_each_state = time_at_each_state[:first_nan].astype(int)

        total_transitions_time = np.sum(time_at_each_state)
        current_time = row["current_time"]
        if (current_time > total_transitions_time) & (
            states[-1] not in terminal_states
        ):
            time_at_each_state = np.append(
                time_at_each_state, current_time - total_transitions_time
            )

        # edge case where final state is not terminal state and there was a last day transition to it
        if (len(states) != len(time_at_each_state)) & (
            states[-1] not in terminal_states
        ):
            time_at_each_state = np.append(time_at_each_state, np.array([1]))

        # bug fix for zero transition times
        time_at_each_state[time_at_each_state == 0] = 1

        total_time = np.sum(time_at_each_state)

        # add id
        sample_id = row["id"]

        # add covariates
        covariates = pd.to_numeric(row[["is_male", "age"]])
        covariates["was_severe"] = 0
        # covariates["cum_hosp_time"] = 0

        if verbose:
            print("\n", sample_id)
            print(states)
            print(time_at_each_state)
            print(total_transitions_time, current_time)
            print(total_time)

        path = PathObject(
            covariates=covariates,
            states=states,
            time_at_each_state=time_at_each_state,
            sample_id=sample_id,
        )
        return path

    age_mapper = {
        "[55.0, 60.0)": 57.5,
        "[75.0, 80.0)": 77.5,
        "[80.0, 105.0)": 92.5,
        "[70.0, 75.0)": 72.5,
        "[45.0, 50.0)": 47.5,
        "[25.0, 30.0)": 27.5,
        "[60.0, 65.0)": 62.5,
        "[35.0, 40.0)": 37.5,
        "[20.0, 25.0)": 22.5,
        "[0.0, 20.0)": 10,
        "[65.0, 70.0)": 67.5,
        "[40.0, 45.0)": 42.5,
        "[50.0, 55.0)": 52.5,
        "[30.0, 35.0)": 32.5,
    }

    sex_mapper = {"Male": 1, "Female": 0}

    invalid_path_ids = []

    df = pd.read_csv(resource_filename("pymsm", "datasets/covid_hosp_data.csv"))
    df["age"] = df["age_group"].map(age_mapper)
    df["is_male"] = df["sex"].map(sex_mapper)

    # rename states
    states_mapper = {0: 0, 16: 1, 23: 2, 4: 3, 5: 4}
    states_labels = {0: "Censored", 1: "OOHQ", 2: "M&M", 3: "Severe", 4: "Deceased"}
    for col in state_cols:
        df[col] = df[col].map(states_mapper).astype(int)

    dataset = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row["id"] in invalid_path_ids:  # invalid paths
            continue
        else:
            dataset.append(parse_row(row, verbose=False))
    return dataset
