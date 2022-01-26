import numpy as np
from pymsm.multi_state_competing_risks_model import PathObject
from typing import List
from seaborn import ecdfplot


def prob_visited_state(paths: List[PathObject], state: int):
    return np.mean([int(state in path.states) for path in paths])


def prob_visited_states(paths: List[PathObject], states: List[int]):
    states = set(states)
    return np.mean([len(states.intersection(set(path.states))) > 0 for path in paths])


def path_total_time_at_states(path: PathObject, states: List[int]):
    # take care and drop terminal states
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


def plot_total_times_ecdf(paths: List[PathObject], states: List[int], ax=None):
    total_times = np.array([path_total_time_at_states(path, states) for path in paths])
    if ax is None:
        fig, ax = plt.subplots()
    ecdfplot(x=total_times, ax=ax)

if __name__ == "__main__":
    pass

