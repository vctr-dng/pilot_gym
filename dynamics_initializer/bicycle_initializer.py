import numpy as np

from components_registry import register
from track import Track


@register("dynamics_initializer/bicycle")
class BicycleInitializer:
    def __init__(self, boundaries: dict, vehicle_params: dict, simulation_params: dict):
        self.vehicle_params = vehicle_params
        self.simulation_params = (
            simulation_params  # useful to add noise or rollout before the actual start
        )
        self.boundaries = boundaries

    def __call__(self, track: Track) -> dict:
        random_index = np.random.randint(0, len(track.reference_path))
        random_lateral_proportion = np.random.uniform(
            self.boundaries["lateral_proportion"]["min"],
            self.boundaries["lateral_proportion"]["max"],
        )

        left_point = track.left_boundaries[random_index]
        right_point = track.right_boundaries[random_index]

        initial_position = left_point + (random_lateral_proportion + 1) / 2 * (
            right_point - left_point
        )
        initial_heading = track.local_heading_map[random_index]
        random_velocity = np.random.uniform(
            self.simulation_params["min_velocity"], self.vehicle_params["max_velocity"]
        )

        initial_state = {
            "x": initial_position[0],
            "y": initial_position[1],
            "heading": initial_heading,
            "steering": 0,
            "slip_angle": 0,
            "velocity": random_velocity,
            "acceleration": 0,
        }

        return initial_state
