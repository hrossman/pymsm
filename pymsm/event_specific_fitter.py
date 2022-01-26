from abc import ABC, abstractmethod
from lifelines import CoxPHFitter
import numpy as np


class EventSpecificFitter(ABC):

    def __init__(self):
        return

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def coefficients(self) -> np.ndarray:
        pass

    @abstractmethod
    def unique_event_times(self) -> np.ndarray:
        pass

    @abstractmethod
    def baseline_hazard(self) -> np.ndarray:
        pass

    @abstractmethod
    def baseline_cumulative_hazard(self) -> np.ndarray:
        pass


class CoxWrapper(EventSpecificFitter):
    def __init__(self):
        super(CoxWrapper, self).__init__()


    def fit(self):
        pass

    def coefficients(self) -> np.ndarray:
        pass

    def unique_event_times(self) -> np.ndarray:
        pass

    def baseline_hazard(self) -> np.ndarray:
        pass

    def baseline_cumulative_hazard(self) -> np.ndarray:
        pass
