from __future__ import annotations

from pathlib import Path

import yaml

from components_registry import register

from .base_track_controller import BaseTrackController
from .track_data import LocalTrackData


@register("track_controller/local_list_controller")
class LocalListController(BaseTrackController):
    def __init__(self, list_path: str | Path):
        track_list = self.from_list_file(list_path)
        super().__init__(track_list)

    @staticmethod
    def from_list_file(list_path: str | Path) -> list[LocalTrackData]:
        if isinstance(list_path, str):
            list_path = Path(list_path)

        if not list_path.exists():
            raise FileNotFoundError(f"Track list file {list_path} does not exist")

        # switch case depending on the file extension
        callable_reader: callable = None
        match list_path.suffix:
            case ".yaml":
                callable_reader = yaml.safe_load
            case _:
                raise ValueError(f"Unsupported file extension {list_path.suffix}")

        with open(list_path, "r") as file:
            list_track_data = callable_reader(file)
            if not isinstance(list_track_data, list):
                raise ValueError("The content of the file is not a list")

        return LocalListController.from_list(list_track_data)

    @staticmethod
    def from_list(l_local_path: list[str | Path]) -> list[LocalTrackData]:
        list_track_data = []
        for path in l_local_path:
            if isinstance(path, str):
                path = Path(path)

            if not path.exists():
                raise FileNotFoundError(f"Track file {path} does not exist")

            list_track_data.append(LocalTrackData(path.stem, path))

        return list_track_data
