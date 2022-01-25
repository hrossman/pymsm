import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymsm.competing_risks_model import CompetingRisksModel
from lifelines import AalenJohansenFitter
from typing import List


def stackplot(
    data,
    order_top: List = None,
    order_bottom: List = None,
    times: np.ndarray = None,
    ax=None,
):
    if times is None:
        times = np.sort(data["time"].unique())

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, times[-1])
        ax.set_ylabel("Probability")
        ax.set_xlabel("t")

    cumulative_densities = [0]
    for status in [1, 2]:
        T = data["time"]
        E = data["status"]
        ajf = AalenJohansenFitter()
        ajf.fit(T, E, event_of_interest=status)
        cumulative_densities.append(ajf.predict(times, interpolate=True))

    cifs_top = []
    for i, failure_type in enumerate(order_top):
        cif = 1 - cumulative_densities[failure_type].values
        times = cumulative_densities[failure_type].index
        if i != 0:
            cif = cif - cifs_top[i - 1]
        ax.plot(times, cif, color="k")
        ax.fill_between(times, 1, cif, alpha=0.8)
        cifs_top.append(cif)

    cifs_bottom = []
    for i, failure_type in enumerate(order_bottom):
        cif = cumulative_densities[failure_type].values
        times = cumulative_densities[failure_type].index
        if i != 0:
            cif = cif + cifs_bottom[i - 1]
        ax.plot(times, cif, color="k")
        ax.fill_between(times, 0, cif, alpha=0.8)
        cifs_top.append(cif)

