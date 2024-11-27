from components_registry import register
from track import Track
from track_controller import BaseTrackController

from .base_track_sampler import BaseTrackSampler


@register("track_sampler/round_robin")
class RoundRobinSampler(BaseTrackSampler):
    def __init__(self, controller: BaseTrackController):
        self.controller: BaseTrackController = controller
        self.track_indexes = self.controller.get_track_indexes()
        self.length = self.controller.length
        self.index: int = 0

    def __call__(self) -> Track:
        track_data = self.controller.get_track_data(self.index)
        track = track_data.get_track()
        self.index = (self.index + 1) % self.length
        return track
