import pandas as pd
from pymsm.utils import get_categorical_columns
from pkg_resources import resource_filename


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


if __name__ == "__main__":
    # competing_risk_dataset, covariate_cols = prep_ebmt_long()
    # print(competing_risk_dataset)
    data = load_aidssi()
    print(data)
