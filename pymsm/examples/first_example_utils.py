from pymsm.multi_state_competing_risks_model import PathObject
import numpy as np
from pandas import Series
from typing import List


def create_one_object(sample_id: int, lambda_param: float) -> PathObject:
    path = PathObject(covariates=Series(dict(zip(['a', 'b'], np.random.normal(size=2)))),
                      sample_id=sample_id)
    current_state = 1
    while current_state != 3:
        path.states.append(current_state)
        transition_to_3 = np.random.binomial(1, 0.5)
        if transition_to_3:
            path.time_at_each_state.append(1)
            current_state = 3
        else:
            path.time_at_each_state.append(np.random.exponential(1/lambda_param))
            current_state = 1 + (current_state % 2)
    path.states.append(3)

    return path


def create_toy_setting_dataset(lambda_param: float) -> List[PathObject]:
    dataset = list()
    for i in range(0, 1000):
        dataset.append(create_one_object(i, lambda_param))
    return dataset
