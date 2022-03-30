from pymsm.multi_state_competing_risks_model import PathObject
import numpy as np
from pandas import Series
from typing import List
import matplotlib.pyplot as plt


def create_one_object(sample_id: int, lambda_param: float) -> PathObject:
    path = PathObject(
        covariates=Series(dict(zip(["a", "b"], np.random.normal(size=2)))),
        sample_id=sample_id,
    )
    current_state = 1
    while current_state != 3:
        path.states.append(current_state)
        transition_to_3 = np.random.binomial(1, 0.5)
        if transition_to_3:
            path.time_at_each_state.append(1)
            current_state = 3
        else:
            path.time_at_each_state.append(np.random.exponential(1 / lambda_param))
            current_state = 1 + (current_state % 2)
    path.states.append(3)

    return path


def create_toy_setting_dataset(lambda_param: float) -> List[PathObject]:
    dataset = list()
    for i in range(0, 1000):
        dataset.append(create_one_object(i, lambda_param))
    return dataset


def plot_total_time_until_terminal_state(all_mcs, true_lambda, ax=None):
    # plot distribution of times to terminal state and compare expected to observed mean time
    ts = [np.sum(mc.time_at_each_state) for mc in all_mcs]
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(ts, bins=np.arange(1, 6.01, 0.25))
    ax.axvline(true_lambda, color="r", alpha=0.8, label="True $\lambda$")
    ax.axvline(np.mean(ts), color="k", alpha=0.8, label="Mean observed $\lambda$")
    ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel("density")
    ax.set_title("Distribution of Total Time Until Terminal State")
