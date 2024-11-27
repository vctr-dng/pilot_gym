from abc import ABC, abstractmethod
from dataclasses import dataclass

from track import Track


@dataclass
class BaseTrackData(ABC):
    name: str

    @abstractmethod
    def get_track(self) -> Track:
        pass


@dataclass
class LocalTrackData(BaseTrackData):
    path: str

    def get_track(self):
        return Track.load(self.path)
