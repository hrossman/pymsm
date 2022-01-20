import numpy as np
from pymsm.competing_risks_model import CompetingRisksModel
from pymsm.examples.crm_example_utils import create_test_data


def test_assert_valid_dataset(crm=None):
    if crm is None:
        crm = CompetingRisksModel()
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
    cox_model1 = crm._fit_event_specific_model(
        event_of_interest=1,
        df=df,
        duration_col="T",
        event_col="transition",
        cluster_col="id",
        verbose=2,
    )

    # Fit second transition
    cox_model2 = crm._fit_event_specific_model(
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
    cox_model = crm._fit_event_specific_model(
        event_of_interest=1,
        df=df,
        duration_col="T",
        event_col="transition",
        cluster_col="id",
        verbose=0,
    )
    attributes_dict = crm.extract_necessary_attributes(cox_model)
    # print(attributes_dict)
    # TODO check lens


def test_all():
    crm = CompetingRisksModel()
    data = create_test_data(N=1_000)
    test_assert_valid_dataset(crm=crm)
    test_break_ties_by_adding_epsilon(crm=crm)
    test_fit_event_specific_model(df=data, crm=None)


if __name__ == "__main__":
    data = create_test_data(N=100)
    test_all()
