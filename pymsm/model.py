import numpy as np


class SinglePatient:
    def __init__(
        self,
        covariates: np.ndarray = None,
        visited_states: np.ndarray = None,
        time_at_each_state: np.ndarray = None,
        id: int = None,
    ) -> None:
        self.covariates = covariates
        self.visited_states = visited_states
        self.time_at_each_state = time_at_each_state
        self.id = id



class MSM:
    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    pass

