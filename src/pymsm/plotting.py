import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymsm.multi_state_competing_risks_model import PathObject
from pymsm.statistics import paths_to_timestep_matrix, get_state_timestep_probs
from lifelines import AalenJohansenFitter
from typing import List, Dict


def stackplot(
    data: pd.DataFrame,
    origin_state: int,
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

    failure_types = np.sort(data[event_col].unique())
    if labels is None:
        labels = dict(zip(failure_types, [str(f) for f in failure_types]))

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, times[-1])
        ax.set_ylabel("Probability", fontsize=fontsize)
        ax.set_xlabel("t", fontsize=fontsize)
        ax.set_title(f"Starting from {labels[origin_state]}", fontsize=fontsize + 4)

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


def stackplot_state_timesteps(
    state_timestep_probs: np.ndarray,
    order_top: List = [],
    order_bottom: List = [],
    times: np.ndarray = None,
    labels: Dict = None,
    fontsize: int = 18,
    ax=None,
):
    failure_types = list(state_timestep_probs.keys())
    if labels is None:
        labels = dict(zip(failure_types, [str(f) for f in failure_types]))

    if times is None:
        times = np.arange(len(state_timestep_probs[failure_types[0]]))

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, times[-1])
        ax.set_ylabel("Probability", fontsize=fontsize)
        ax.set_xlabel("t", fontsize=fontsize)

    cifs_top = []
    for i, failure_type in enumerate(order_top):
        color = f"C{failure_type}"
        cif = 1 - state_timestep_probs[failure_type]
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
        cif = state_timestep_probs[failure_type]
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


def stackplot_state_timesteps_from_paths(
    paths: List[PathObject],
    max_timestep: int,
    order_top: List,
    order_bottom: List,
    labels: Dict = None,
    ax=None,
):
    timestep_matrix = paths_to_timestep_matrix(paths, max_timestep)
    state_timestep_probs = get_state_timestep_probs(timestep_matrix)
    stackplot_state_timesteps(
        state_timestep_probs, order_top, order_bottom, labels=labels, ax=ax
    )
