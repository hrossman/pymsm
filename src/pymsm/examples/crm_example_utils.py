import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def stack_plot(data):
    state0 = []
    state1 = []
    state2 = []
    # times = np.arange(data['T'].max(), dtype=int)
    times = np.arange(10, dtype=int)

    for t in times:
        state0.append((data["T"] > t).sum())
        state1.append(((data["transition"] == 1) & (data["T"] <= t)).sum())
        state2.append(((data["transition"] == 2) & (data["T"] <= t)).sum())

    cifs = [state1, state2, state0]
    labels = ["B", "C", "A"]
    ax = plot_stackplot(times, cifs, labels, ax=None)
    ax.set_ylim(0, len(data))
    plt.show()
