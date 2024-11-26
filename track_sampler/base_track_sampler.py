from abc import ABC, abstractmethod
from typing import Any

from track import Track
from track_controller import BaseTrackController


class BaseTrackSampler(ABC):
    def __init__(self, controller: BaseTrackController):
        self.controller = controller

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Track:
        pass
