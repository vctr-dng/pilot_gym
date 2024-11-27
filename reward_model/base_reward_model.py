from abc import ABC, abstractmethod

import numpy as np


class BaseRewardModel(ABC):
    def __init__(self, coefficients: dict):
        self.coefficients = coefficients

    @abstractmethod
    def __call__(
        self, action: dict, state_observation: dict, track_observation: np.array
    ) -> float:
        pass
