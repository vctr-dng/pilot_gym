from abc import ABC, abstractmethod

import numpy as np

from track import Track


class TrackGenerator(ABC):
    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate

    def generate_straight(self, start_point: np.array, end_point: np.array) -> np.array:
        # euclidian distance between start and end points
        distance = np.linalg.norm(end_point - start_point)

        straight = np.column_stack(
            (
                np.linspace(
                    start_point[0], end_point[0], int(distance * self.sampling_rate)
                ),
                np.linspace(
                    start_point[1], end_point[1], int(distance * self.sampling_rate)
                ),
            )
        )[:-1]

        return straight

    def generate_corner(
        self, center: np.array, radius: float, start_angle: float, end_angle: float
    ) -> np.array:
        num_points = int(np.abs(end_angle - start_angle) * radius * self.sampling_rate)

        sampled_angles = np.linspace(start_angle, end_angle, num_points)

        corner = np.column_stack(
            (
                center[0] + radius * np.cos(sampled_angles),
                center[1] + radius * np.sin(sampled_angles),
            )
        )

        return corner

    @abstractmethod
    def generate_track(self) -> Track:
        pass
