from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree


class Track:
    # TODO: move processing to another class

    right_boundaries: np.ndarray
    left_boundaries: np.ndarray
    center_line: np.ndarray
    reference_path: np.ndarray
    progress_map: np.ndarray

    def __init__(
        self,
        right_boundaries: np.ndarray,
        left_boundaries: np.ndarray,
        center_line: np.ndarray,
        reference_path: np.ndarray = None,
        closed_loop: bool = True,
    ):
        # Process the boundaries and centerline to have equal points
        self.right_boundaries, self.left_boundaries, self.center_line = (
            self._process_track_points(right_boundaries, left_boundaries, center_line)
        )
        self.reference_path = (
            self.center_line if reference_path is None else reference_path
        )
        self.closed_loop = closed_loop
        self.progress_map = self.get_progress_map(self.reference_path)
        self.local_heading_map = self.get_local_heading_map(
            self.reference_path, self.closed_loop
        )

    @staticmethod
    def get_progress_map(reference_path: np.ndarray) -> np.ndarray:
        # Calculate the cumulative distance along the centerline
        distances = np.cumsum(
            np.sqrt(np.sum(np.diff(reference_path, axis=0) ** 2, axis=1))
        )
        distances = np.insert(distances, 0, 0)
        return distances

    @staticmethod
    def get_local_heading_map(
        reference_path: np.ndarray, closed_loop: bool
    ) -> np.ndarray:
        # Calculate the local heading angle at each point
        headings = np.arctan2(
            np.diff(reference_path[:, 1]), np.diff(reference_path[:, 0])
        )

        if closed_loop:
            # calculate the heading of the last point by using the first point as
            # the point after the last one
            headings = np.append(
                headings,
                np.arctan2(
                    reference_path[0, 1] - reference_path[-1, 1],
                    reference_path[0, 0] - reference_path[-1, 0],
                ),
            )
        else:
            # Repeat the last heading for the last point
            headings = np.append(headings, headings[-1])

        return headings

    def _process_track_points(
        self,
        right_boundaries: np.ndarray,
        left_boundaries: np.ndarray,
        center_line: np.ndarray,
    ):
        """
        Project each resampled centerline point onto the left and right boundaries
        by finding the intersection of the normal vector at each centerline point with
        the boundaries using KD-Trees.
        """
        # Step 1: Define the number of points for resampling (use the centerline length)
        num_points = len(center_line)

        # Step 2: Resample the centerline to have 'num_points' equally spaced points
        center_line_resampled = self._resample_line(center_line, num_points)

        # Step 3: Project each centerline point onto the left and right boundaries
        # using normal vectors
        left_boundaries_resampled = self._project_onto_boundaries(
            center_line_resampled, left_boundaries
        )
        right_boundaries_resampled = self._project_onto_boundaries(
            center_line_resampled, right_boundaries
        )

        return (
            right_boundaries_resampled,
            left_boundaries_resampled,
            center_line_resampled,
        )

    def _resample_line(self, line: np.ndarray, num_points: int) -> np.ndarray:
        # Calculate cumulative distances along the line
        distances = np.cumsum(np.sqrt(np.sum(np.diff(line, axis=0) ** 2, axis=1)))
        distances = np.insert(
            distances, 0, 0
        )  # Add zero distance for the starting point

        # Create new set of equally spaced distances along the total length
        uniform_distances = np.linspace(0, distances[-1], num_points)

        # Interpolate line coordinates based on uniform distances
        resampled_x = np.interp(uniform_distances, distances, line[:, 0])
        resampled_y = np.interp(uniform_distances, distances, line[:, 1])

        return np.vstack((resampled_x, resampled_y)).T

    def _calculate_normal_vector(self, point1, point2):
        # Calculate the normal vector for the segment defined by point1 and point2.

        # Tangent vector: direction along the track
        tangent_vector = point2 - point1
        tangent_vector /= np.linalg.norm(tangent_vector)  # Normalize the tangent vector

        # Normal vector: perpendicular to the tangent (rotate tangent by 90 degrees)
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])

        return normal_vector

    def _find_intersection_point(self, kd_tree: KDTree, origin, direction):
        # Find the intersection of a normal vector with a boundary using KD-Tree.

        # Search for points within a distance from the normal vector's origin
        search_radius = 10.0  # Define a search radius for boundary intersection
        indices = kd_tree.query_ball_point(origin, search_radius)

        # If no points are found, return None
        if not indices:
            print("No nearby boundary points found.")
            return None

        # Among the nearby boundary points, find the one closest to the normal line
        closest_point = None
        min_distance = np.inf

        for idx in indices:
            boundary_point = kd_tree.data[idx]
            # Project the boundary point onto the normal vector
            projection_length = np.dot(boundary_point - origin, direction)
            projection_point = origin + projection_length * direction

            # Compute the distance between the boundary point and the projection
            distance = np.linalg.norm(boundary_point - projection_point)

            if distance < min_distance:
                min_distance = distance
                closest_point = boundary_point

        return closest_point

    def _project_onto_boundaries(
        self, centerline: np.ndarray, boundaries: np.ndarray
    ) -> np.ndarray:
        # Project centerline points onto the boundary using normal vectors and KD-Trees.

        # Step 1: Build a KD-Tree for the boundary points
        kd_tree = KDTree(boundaries)

        # Step 2: Iterate over each centerline point and project it onto the boundary
        boundary_projections = []

        for i in range(len(centerline) - 1):
            # Get the current point and next point to compute the tangent
            # and normal vector
            center_point = centerline[i]
            next_point = centerline[i + 1]

            # Calculate the normal vector at the current centerline point
            normal_vector = self._calculate_normal_vector(center_point, next_point)

            # Find the intersection of the normal vector with the boundary
            intersection_point = self._find_intersection_point(
                kd_tree, center_point, normal_vector
            )

            # Append the intersection point (or handle edge case if None)
            if intersection_point is not None:
                boundary_projections.append(intersection_point)
            else:
                # Handle cases where no intersection is found (e.g., extrapolate from
                # previous points)
                print(f"No intersection found for centerline point {i}.")
                boundary_projections.append(
                    center_point
                )  # Fallback to centerline point

        # Handle the last point (same approach as the previous one)
        boundary_projections.append(
            boundaries[-1]
        )  # Fallback to the last boundary point

        return np.array(boundary_projections)

    def save(self, path: str | Path):
        np.savez(
            path,
            right_boundaries=self.right_boundaries,
            left_boundaries=self.left_boundaries,
            center_line=self.center_line,
            reference_path=self.reference_path,
        )

    @staticmethod
    def load(path: str | Path) -> Track:
        data = np.load(path)

        expected_keys = [
            "right_boundaries",
            "left_boundaries",
            "center_line",
            "reference_path",
        ]

        for key in expected_keys:
            if key not in data:
                raise ValueError(f"Expected key {key} not found in the loaded data.")

        return Track(
            data["right_boundaries"],
            data["left_boundaries"],
            data["center_line"],
            data["reference_path"],
        )

    def plot(self):
        plt.plot(self.center_line[:, 0], self.center_line[:, 1], label="center")
        plt.plot(self.left_boundaries[:, 0], self.left_boundaries[:, 1], label="left")
        plt.plot(
            self.right_boundaries[:, 0], self.right_boundaries[:, 1], label="right"
        )

        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()
