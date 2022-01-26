import numpy as np
from pymsm.statistics import *


def main():
    states = [1, 2, 3]
    time_at_each_state = [1, 2, 3]
    max_timestep = 5
    states_at_timestep = make_states_at_timestep_array(
        states, time_at_each_state, max_timestep, 0, True
    )
    assert (states_at_timestep == np.array([1, 2, 2, 3, 3])).all()

    states = [1, 2, 3]
    time_at_each_state = [1, 2, 3]
    max_timestep = 7
    states_at_timestep = make_states_at_timestep_array(
        states, time_at_each_state, max_timestep, 0, True
    )
    assert (states_at_timestep == np.array([1, 2, 2, 3, 3, 3, 0])).all()

    states = [1]
    time_at_each_state = [3]
    max_timestep = 5
    states_at_timestep = make_states_at_timestep_array(
        states, time_at_each_state, max_timestep, 0, True
    )
    assert (states_at_timestep == np.array([1, 1, 1, 0, 0])).all()

    states = [1, 2, 5]
    time_at_each_state = [1, 2]
    max_timestep = 7
    states_at_timestep = make_states_at_timestep_array(
        states, time_at_each_state, max_timestep, 0, True
    )
    assert (states_at_timestep == np.array([1, 2, 2, 5, 5, 5, 5])).all()

    states = [1, 2, 3]
    time_at_each_state = [1.4, 1.7, 2.6]
    max_timestep = 7
    states_at_timestep = make_states_at_timestep_array(
        states, time_at_each_state, max_timestep, 0, True
    )
    assert (states_at_timestep == np.array([1, 2, 2, 3, 3, 3, 0])).all()


if __name__ == "__main__":
    main()
