from abc import ABC, abstractmethod


class DynamicModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def state(self):
        pass
