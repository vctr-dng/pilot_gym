from abc import ABC

from .track_data import BaseTrackData


class BaseTrackController(ABC):
    def __init__(self, track_collection):
        if not hasattr(track_collection, "__getitem__"):
            raise ValueError("track_collection must support indexing")

        self.track_collection = track_collection
        self.length = len(track_collection)

    def get_track_indexes(self) -> list:
        return list(iter(self.track_collection))

    def get_track_data(self, index) -> BaseTrackData:
        track_data: BaseTrackData = self.track_collection[index]
        return track_data
