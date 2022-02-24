import numpy as np
import pandas as pd
from pymsm.multi_state_competing_risks_model import PathObject
from typing import List, Dict
from collections import Counter


def get_path_frequencies(paths: List[PathObject], states_labels: Dict = None):
    """Get a dictionary of path frequencies for a given list of paths"""
    states_list = []
    for path in paths:
        states_list.append(path.states)

    # Change from numbers to labels
    if states_labels is not None:
        states_list = [[states_labels[y] for y in x] for x in states_list]

    counter = Counter(tuple(x) for x in states_list)
    path_freqs = {
        str(k)
        .replace(", ", "->")
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
        .replace(",", ""): v
        for k, v in counter.items()
    }
    return pd.Series(path_freqs).sort_values(ascending=False)


def prob_visited_state(paths: List[PathObject], state: int):
    return np.mean([int(state in path.states) for path in paths])


def prob_visited_states(paths: List[PathObject], states: List[int]):
    states = set(states)
    return np.mean([len(states.intersection(set(path.states))) > 0 for path in paths])


def path_total_time_at_states(path: PathObject, states: List[int]):
    # drop terminal states
    num_nonterminal_states = len(path.time_at_each_state)
    nonterminal_path_states = path.states[:num_nonterminal_states]
    idx = np.isin(nonterminal_path_states, states)
    relevant_times = np.array(path.time_at_each_state)[idx]
    return np.sum(relevant_times)


def stats_total_time_at_states(
    paths: List[PathObject], states: List[int], quantiles=[0.1, 0.25, 0.75, 0.9]
):
    total_times = [path_total_time_at_states(path, states) for path in paths]
    stats = {
        "time_in_state_mean": np.mean(total_times),
        "time_in_state_std": np.std(total_times),
        "time_in_state_median": np.median(total_times),
        "time_in_state_min": np.min(total_times),
        "time_in_state_max": np.max(total_times),
    }
    for q in quantiles:
        stats[f"time_in_state_quantile_{q}"] = np.quantile(total_times, q)
    return stats


def make_states_at_timestep_array(
    states: List,
    time_at_each_state: List,
    max_timestep: int,
    start_time: float = 0,
    rounding: bool = True,
):
    time_at_each_state = np.array(time_at_each_state) - start_time

    # rounding procedure, works on cumsum
    if rounding:
        time_at_each_state = np.diff(
            np.round(np.cumsum(time_at_each_state)), prepend=0
        ).astype(int)

    # only repeat non-terminal states
    num_nonterminal_states = len(time_at_each_state)
    if num_nonterminal_states != len(states):
        ended_with_terminal_state = True
        nonterminal_path_states = states[:num_nonterminal_states]
        terminal_path_state = states[num_nonterminal_states]
        # make repeated array
        states_at_timestep_full = np.repeat(nonterminal_path_states, time_at_each_state)
        # add terminal state at the end
        states_at_timestep_full = np.concatenate(
            [states_at_timestep_full, np.array([terminal_path_state])]
        )
    else:
        ended_with_terminal_state = False
        nonterminal_path_states = states
        # make repeated array
        states_at_timestep_full = np.repeat(nonterminal_path_states, time_at_each_state)

    # clip to inculde only max time step
    states_at_timestep = states_at_timestep_full[:max_timestep]
    # fill up to max timestep
    if len(states_at_timestep) < max_timestep:
        fill_shape = max_timestep - len(states_at_timestep)
        if ended_with_terminal_state:
            fill_value = states_at_timestep[-1]
        else:
            fill_value = 0
        states_at_timestep = np.concatenate(
            [states_at_timestep, np.full(fill_shape, fill_value)]
        )
    # print(states_at_timestep)

    return states_at_timestep


def path_to_timestep_array(
    path: PathObject, max_timestep: int, start_time: float = 0, rounding: bool = True
):
    return make_states_at_timestep_array(
        path.states, path.time_at_each_state, max_timestep, start_time, rounding
    )


def paths_to_timestep_matrix(
    paths: List[PathObject],
    max_timestep: int,
    start_time: float = 0,
    rounding: bool = True,
):
    return np.concatenate(
        [
            [path_to_timestep_array(path, max_timestep, start_time, rounding)]
            for path in paths
        ]
    )


def get_state_timestep_probs(timestep_matrix: np.ndarray) -> Dict:
    state_timestep_probs = {}
    censored_counts = 0
    for state in np.unique(timestep_matrix):
        mask = timestep_matrix == state
        counts = mask.sum(axis=0)  # sum in each timestep
        if state == 0:
            censored_counts = mask.sum(axis=0)
            continue
        probs = counts / (
            len(timestep_matrix) - censored_counts
        )  # exclude censored counts
        state_timestep_probs[state] = probs

    return state_timestep_probs


if __name__ == "__main__":
    pass
