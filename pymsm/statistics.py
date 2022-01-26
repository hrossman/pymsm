from pathlib import Path
import numpy as np
from pymsm.multi_state_competing_risks_model import PathObject
from typing import List


def prob_visited_state(paths: List[PathObject], state: int):
    return np.mean([int(state in path.states) for path in paths])


def prob_visited_states(paths: List[PathObject], states: List):
    states = set(states)
    return np.mean([len(states.intersection(set(path.states))) > 0 for path in paths])


# time_at_hospital = function(monte_carlo_run) {
#   states = monte_carlo_run$states
#   time_at_each_state = monte_carlo_run$time_at_each_state
  
#   if (length(states) > length(time_at_each_state)){
#     states = head(states, -1)
#   }
  
#   return(sum(time_at_each_state[states != RECOVERED_OR_OOHQ]))
# }



if __name__ == "__main__":
    a = [1, 2]
    times = [3, 2]

    arr = np.repeat(a, times)
    print(arr)

