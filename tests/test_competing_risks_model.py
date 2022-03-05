import numpy as np
import pandas as pd
from pymsm.competing_risks_model import CompetingRisksModel


def create_test_data(N=100):
    data = pd.DataFrame(
        {
            "id": np.arange(N),
            "sex": np.random.binomial(n=1, p=0.5, size=N),
            "age": np.round(np.random.uniform(20, 80, size=N)),
        }
    )
    data["source"] = 0
    data["target"] = np.random.choice(a=[1, 2], size=N, p=[0.5, 0.5])
    data["transition"] = data["target"]
    data.loc[data["transition"] == 1, "T"] = 1 + np.round(
        np.random.exponential(scale=2, size=np.sum(data["transition"] == 1))
    )
    data.loc[data["transition"] == 2, "T"] = 1 + np.round(
        np.random.exponential(scale=4, size=np.sum(data["transition"] == 2))
    )
    data.drop(["source", "target"], axis=1, inplace=True)
    return data


def test_assert_valid_dataset(data=None, crm=None):
    if crm is None:
        crm = CompetingRisksModel()
    if data is None:
        data = create_test_data(N=1_000)
    crm.assert_valid_dataset(df=data, duration_col="T", event_col="transition")


def test_break_ties_by_adding_epsilon(crm=None):
    if crm is None:
        crm = CompetingRisksModel()
    t = np.array([0, 0, 1, 2, 2, 3, 4, 4])
    epsilon_min = 0
    epsilon_max = 0.1
    t_after = crm.break_ties_by_adding_epsilon(t, epsilon_min, epsilon_max)
    correct_t_after = [
        0.0,
        0.09507143,
        1.0,
        2.07319939,
        2.05986585,
        3.0,
        4.01560186,
        4.01559945,
    ]
    np.testing.assert_almost_equal(t_after, correct_t_after, decimal=5)


def test_fit_event_specific_model(df=None, crm=None):
    if crm is None:
        crm = CompetingRisksModel()
    if df is None:
        df = create_test_data(N=1_000)

    # Fit first transition
    cox_model1 = crm.fit_event_specific_model(
        event_of_interest=1,
        df=df,
        duration_col="T",
        event_col="transition",
        cluster_col="id",
        verbose=2,
    )

    # Fit second transition
    cox_model2 = crm.fit_event_specific_model(
        event_of_interest=2,
        df=df,
        duration_col="T",
        event_col="transition",
        cluster_col="id",
        verbose=2,
    )


def test_extract_necessary_attributes(crm=None, df=None):
    if crm is None:
        crm = CompetingRisksModel()
    if df is None:
        df = create_test_data(N=1_000)
    cox_model = crm.fit_event_specific_model(
        event_of_interest=1,
        df=df,
        duration_col="T",
        event_col="transition",
        cluster_col="id",
        verbose=0,
    )


def test_all():
    crm = CompetingRisksModel()
    data = create_test_data(N=1_000)
    test_assert_valid_dataset(crm=crm)
    test_break_ties_by_adding_epsilon(crm=crm)
    test_fit_event_specific_model(df=data, crm=None)
