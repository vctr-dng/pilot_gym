from __future__ import annotations

from pathlib import Path

from components_registry import register

from .base_track_controller import BaseTrackController
from .track_data import LocalTrackData


@register("track_controller/local_list_controller")
class LocalListController(BaseTrackController):
    def __init__(self, track_list: list[LocalTrackData]):
        super().__init__(track_list)

    @staticmethod
    def from_list(l_local_path: list[str | Path]) -> LocalListController:
        list_track_data = []
        for path in l_local_path:
            if isinstance(path, str):
                path = Path(path)

            if not path.exists():
                raise FileNotFoundError(f"Track file {path} does not exist")
            list_track_data.append(LocalTrackData(path))

        return LocalListController(list_track_data)
