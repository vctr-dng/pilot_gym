from typing import Tuple

import numpy as np
from scipy.spatial import KDTree

from components_registry import register
from track import Track


@register("track_observer/simple_track_observer")
class SimpleTrackObserver:
    def __init__(self, track: Track, observed_state: list, track_description: dict):
        self.track: Track = track
        self.track_tree: KDTree = KDTree(track.reference_path)
        self.observed_state: list = observed_state
        self.track_description: dict = track_description
        self.observation_size: int = (
            len(observed_state) + 3 * 2 * track_description["num_points"]
        )

    def __call__(self, observation_point: np.ndarray, heading: float) -> np.ndarray:
        relative_points = self.get_relative_points(
            observation_point,
            heading,
            self.track_description["num_points"],
            self.track_description["stride"],
        )
        additional_state = np.empty(len(self.observed_state))
        for i in range(len(self.observed_state)):
            state: float = None
            match self.observed_state[i]:
                case "lateral_proportion":
                    state = self.get_lateral_proportion(observation_point)
                case "relative_heading":
                    state = self.get_relative_heading(observation_point, heading)
                case _:
                    raise ValueError(
                        f"Unknown observed track state: {self.observed_state[i]}"
                    )
            additional_state[i] = state

        return np.hstack((additional_state, relative_points))

    def get_closest_info(self, point: np.ndarray) -> Tuple[float, int]:
        """
        Get the distance to the closest point on the reference path and the index of
        that point.

        :param point: A NumPy array representing the coordinates of the point.
        :type point: np.ndarray
        :return: A tuple containing the distance to the closest point and the index of
        the closest point.
        :rtype: Tuple[float, int]
        """

        distance, index = self.track_tree.query(point)
        return distance, index

    def get_closest_point(self, point: np.ndarray) -> np.ndarray:
        """
        Get the closest point on the reference path to the given point.

        :param point: A NumPy array representing the coordinates of the point.
        :type point: np.ndarray
        :return: A NumPy array representing the closest point on the track.
        :rtype: np.ndarray
        """

        _, index = self.get_closest_info(point)
        return self.track.reference_path[index]

    def get_closest_index(self, point: np.ndarray) -> int:
        """
        Get the index of the closest point on the reference path to the given point.

        :param point: A NumPy array representing the coordinates of the point.
        :type point: np.ndarray
        :return: The index of the closest point on the reference path.
        :rtype: int
        """

        _, index = self.get_closest_info(point)
        return index

    def get_relative_heading(self, point: np.ndarray, heading: float) -> float:
        """
        Get the heading angle of the given point relative to the heading of closest
        point on the reference path.

        :param point: A NumPy array representing the coordinates of the point.
        :type point: np.ndarray
        :param heading: The heading angle at the given point.
        :type heading: float
        :return: The relative heading angle of the point along the reference path.
        :rtype: float
        """

        _, index = self.get_closest_info(point)
        return self.track.local_heading_map[index] - heading

    def get_lateral_proportion(self, point: np.ndarray) -> float:
        """
        Get the lateral proportion of the given point on the track. The proportion is
        normalized between -1 and 1, where -1 corresponds
        to the left boundary, 1 corresponds to the right boundary, and 0 is the center
        of the track.

        :param point: A NumPy array representing the coordinates of the point.
        :type point: np.ndarray
        :return: A float representing the lateral proportion of the point.
        :rtype: float
        """

        _, index = self.get_closest_info(point)

        left_boundary = self.track.left_boundaries[index]
        right_boundary = self.track.right_boundaries[index]

        # Step 3: Calculate the local vector from left to right boundary
        boundary_vector = right_boundary - left_boundary

        # Step 4: Project the point onto the boundary vector to find
        # its relative position
        point_vector = point - left_boundary
        projection_length = np.dot(point_vector, boundary_vector) / np.linalg.norm(
            boundary_vector
        )

        # Step 5: Calculate the proportion of the point along the left-right boundary
        total_boundary_length = np.linalg.norm(boundary_vector)
        proportion = projection_length / total_boundary_length

        # Step 6: Return the proportion normalized between [-1, 1]
        return 2 * proportion - 1  # Scale proportion to range [-1, 1

    @staticmethod
    def get_relative_position(
        origin: np.ndarray, heading: float, points: np.ndarray
    ) -> np.ndarray:
        """
        Express the points in the local coordinate system based on the given origin and
        heading.

        :param origin: A NumPy array representing the coordinates of the origin point.
        :type origin: np.ndarray
        :param heading: The heading angle of the origin in radians.
        :type heading: float
        :param points: A NumPy array representing the global coordinates of the points.
        :type points: np.ndarray
        :return: A NumPy array of the points in the local coordinate system.
        :rtype: np.ndarray
        """

        # Translation
        relative_points = points - origin

        # Rotation **- heading angle**
        rotation_matrix = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        return np.dot(relative_points, rotation_matrix)

    def get_relative_points(
        self, origin: np.ndarray, heading: float, num_points: int, stride: int = 1
    ) -> np.ndarray:
        """
        Get the relative positions of points on the track to the given origin.
        This function returns the relative positions of the track's reference points,
        left boundaries, and right boundaries with respect to the given origin and
        heading. The number of points and stride between points can be controlled.

        :param origin: A NumPy array representing the coordinates of the origin point.
        :type origin: np.ndarray
        :param heading: The heading angle of the origin in radians.
        :type heading: float
        :param num_points: The number of points to return.
        :type num_points: int
        :param stride: The stride to use when selecting points (default is 1).
        :type stride: int
        :return: A NumPy array containing the relative reference points,
        left boundaries, and right boundaries.
        :rtype: np.ndarray
        """

        _, index = self.get_closest_info(origin)

        wrapped_indices = np.arange(index, index + num_points * stride, stride) % len(
            self.track.reference_path
        )

        relative_reference = self.get_relative_position(
            origin,
            heading,
            self.track.reference_path[wrapped_indices],
        )

        relative_left_boundaries = self.get_relative_position(
            origin,
            heading,
            self.track.left_boundaries[wrapped_indices],
        )

        relative_right_boundaries = self.get_relative_position(
            origin,
            heading,
            self.track.right_boundaries[wrapped_indices],
        )

        return np.hstack(
            (relative_reference, relative_left_boundaries, relative_right_boundaries)
        ).flatten()
