from pymsm.statistics import *


def test_make_states_at_timestep_array():
    states = [1, 2, 3]
    time_at_each_state = [1, 2, 3]
    max_timestep = 5
    states_at_timestep = make_states_at_timestep_array(
        states, time_at_each_state, max_timestep, 0, True
    )
    np.testing.assert_equal(states_at_timestep, np.array([1, 2, 2, 3, 3]))

    states = [1, 2, 3]
    time_at_each_state = [1, 2, 3]
    max_timestep = 7
    states_at_timestep = make_states_at_timestep_array(
        states, time_at_each_state, max_timestep, 0, True
    )
    np.testing.assert_equal(states_at_timestep, np.array([1, 2, 2, 3, 3, 3, 0]))

    states = [1]
    time_at_each_state = [3]
    max_timestep = 5
    states_at_timestep = make_states_at_timestep_array(
        states, time_at_each_state, max_timestep, 0, True
    )
    np.testing.assert_equal(states_at_timestep, np.array([1, 1, 1, 0, 0]))

    states = [1, 2, 5]
    time_at_each_state = [1, 2]
    max_timestep = 7
    states_at_timestep = make_states_at_timestep_array(
        states, time_at_each_state, max_timestep, 0, True
    )
    np.testing.assert_equal(states_at_timestep, np.array([1, 2, 2, 5, 5, 5, 5]))

    states = [1, 2, 3]
    time_at_each_state = [1.4, 1.7, 2.6]
    max_timestep = 7
    states_at_timestep = make_states_at_timestep_array(
        states, time_at_each_state, max_timestep, 0, True
    )
    np.testing.assert_equal(states_at_timestep, np.array([1, 2, 2, 3, 3, 3, 0]))


def test_paths_to_timestep_matrix():
    test_paths = [
        PathObject(states=[1, 2, 3], time_at_each_state=[1, 2, 1]),
        PathObject(states=[1, 2, 3], time_at_each_state=[2, 2, 1]),
        PathObject(states=[1, 5], time_at_each_state=[2]),
    ]
    true_timestep_matrix = np.array([[1, 2, 2, 3, 0], [1, 1, 2, 2, 3], [1, 1, 5, 5, 5]])
    max_timestep = 5
    timestep_matrix = paths_to_timestep_matrix(test_paths, max_timestep)
    np.testing.assert_equal(timestep_matrix, true_timestep_matrix)

    get_state_timestep_probs(timestep_matrix)

    prob_visited_state(paths=test_paths, state=2)
