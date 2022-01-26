from tkinter import font
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytest import fail
from pymsm.competing_risks_model import CompetingRisksModel
from lifelines import AalenJohansenFitter
from typing import List, Dict


def stackplot(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    order_top: List = [],
    order_bottom: List = [],
    times: np.ndarray = None,
    labels: Dict = None,
    fontsize: int = 18,
    ax=None,
):
    if times is None:
        times = np.sort(data[duration_col].unique())

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, times[-1])
        ax.set_ylabel("Probability", fontsize=fontsize)
        ax.set_xlabel("t", fontsize=fontsize)

    failure_types = np.sort(data[event_col].unique())
    if labels is None:
        labels = dict(zip(failure_types, [str(f) for f in failure_types]))

    cumulative_densities = dict(zip(failure_types, ([0] * len(failure_types))))
    for failure_type in failure_types:
        T = data[duration_col]
        E = data[event_col]
        ajf = AalenJohansenFitter()
        ajf.fit(T, E, event_of_interest=failure_type)
        cumulative_densities[failure_type] = ajf.predict(times, interpolate=True)

    cifs_top = []
    for i, failure_type in enumerate(order_top):
        color = f"C{failure_type}"
        cif = 1 - cumulative_densities[failure_type].values
        if i == 0:
            ax.fill_between(times, 1, cif, alpha=0.8, color=color)
            ax.text(
                x=times[-1] * 1.02,
                y=(cif[-1] + (1 - cif[-1]) / 2),
                s=labels[failure_type],
                fontsize=fontsize,
                color=color,
            )
        else:
            cif = cif - cifs_top[i - 1]
            ax.fill_between(times, cifs_top[i - 1], cif, alpha=0.8, color=color)
            ax.text(
                x=times[-1] * 1.02,
                y=(cif[-1] + (cifs_top[i - 1][-1] - cif[-1]) / 2),
                s=labels[failure_type],
                fontsize=fontsize,
                color=color,
            )
        ax.plot(times, cif, color="k")
        cifs_top.append(cif)

    cifs_bottom = []
    for i, failure_type in enumerate(order_bottom):
        color = f"C{failure_type}"
        cif = cumulative_densities[failure_type].values
        if i == 0:
            ax.fill_between(times, 0, cif, alpha=0.8, color=color)
            ax.text(
                x=times[-1] * 1.02,
                y=((cif[-1]) / 2),
                s=labels[failure_type],
                fontsize=fontsize,
                color=color,
            )
        else:
            cif = cif + cifs_bottom[i - 1]
            ax.fill_between(
                times,
                cifs_bottom[i - 1],
                cif,
                alpha=0.8,
                label=labels[failure_type],
                color=color,
            )
            ax.text(
                x=times[-1] * 1.02,
                y=(cif[-1] + (cifs_bottom[i - 1][-1] - cif[-1]) / 2),
                s=labels[failure_type],
                fontsize=fontsize,
                color=color,
            )
        ax.plot(times, cif, color="k")
        cifs_bottom.append(cif)

