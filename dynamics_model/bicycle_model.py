import numpy as np

from components_registry import register
from dynamics_model import DynamicModel


@register("dynamic_model/bicycle")
class BicycleModel(DynamicModel):
    actions = [
        "throttle",
        "braking",
        "steering_rate",
    ]

    def __init__(self, vehicle_params: dict, simulation_params: dict):
        self.x = 0  # m
        self.y = 0  # m
        self.heading = 0  # radian
        self.steering = 0  # radian
        self.slip_angle = 0  # radian
        self.velocity = 0  # m/s
        self.v_x = 0
        self.v_y = 0
        self.acceleration = 0  # m/s^2

        self.wheelbase = vehicle_params["wheelbase"]
        self.rear_wheel_to_center = vehicle_params["rear_wheel_to_center"]
        self.front_wheel_to_center = self.wheelbase - self.rear_wheel_to_center
        self.max_acceleration = vehicle_params["max_acceleration"]
        self.min_acceleration = np.abs(vehicle_params["min_acceleration"])
        self.max_velocity = vehicle_params["max_velocity"]
        self.lock_to_lock_steering = vehicle_params["lock_to_lock_steering"]
        self.mass = 1000

        # dynamic bicycle model parameters
        self.moment_of_inertia = vehicle_params["moment_of_inertia"]  # kg·m²
        self.tire_stiffness_front = vehicle_params["tire_stiffness_front"]  # N/rad
        self.tire_stiffness_rear = vehicle_params["tire_stiffness_rear"]  # N/rad

        self.max_tire_force_front = (
            self.mass * 9.81 * (self.rear_wheel_to_center / self.wheelbase)
        )
        self.max_tire_force_rear = (
            self.mass * 9.81 * (self.front_wheel_to_center / self.wheelbase)
        )

        self.alpha_front = 0
        self.alpha_rear = 0
        self.Fy_front = 0
        self.Fy_rear = 0

        self.acceleration = 0  # m/s^2, longitudinal acceleration
        self.heading_dot = 0
        self.heading_dot_dot = 0
        self.x_dot = 0
        self.y_dot = 0
        self.v_x_dot = 0
        self.v_y_dot = 0

        self.sim_dt = simulation_params["dt"]
        self.sub_steps = 10
        self.dt = self.sim_dt / self.sub_steps

    def reset(self, initial_state: dict):
        self.verify_initial_state(initial_state)

        self.x = initial_state["x"]
        self.y = initial_state["y"]
        self.heading = initial_state["heading"]
        self.steering = initial_state["steering"]
        self.slip_angle = initial_state["slip_angle"]
        self.velocity = initial_state["velocity"]
        self.v_x = self.velocity
        self.v_y = 0
        self.heading_dot_dot = 0
        self.heading_dot = 0
        self.acceleration = initial_state["acceleration"]
        self.v_x_dot = self.acceleration
        self.v_y_dot = 0
        self.x_dot = 0
        self.y_dot = 0

    def step(self, action: dict):
        self.verify_action(action)
        throttle, braking, steering_rate = (
            action["throttle"],
            action["braking"],
            action["steering_rate"],
        )

        self.acceleration = (
            throttle * self.max_acceleration - braking * self.min_acceleration
        )
        self.acceleration = np.clip(
            self.acceleration, -self.min_acceleration, self.max_acceleration
        )

        if self.velocity >= self.max_velocity and self.acceleration > 0:
            self.acceleration = 0

        # steering_rate is actually used as an angle
        self.steering = steering_rate * self.lock_to_lock_steering / 2
        self.steering = np.clip(
            self.steering,
            -self.lock_to_lock_steering / 2,
            self.lock_to_lock_steering / 2,
        )

        for _ in range(self.sub_steps):
            self.update_dynamic()

    def update_dynamic(self):
        self.alpha_front = self.steering - np.arctan2(
            self.v_y + self.front_wheel_to_center * self.heading_dot,
            max(self.v_x, 1e-3),  # Prevent division by zero
        )
        self.alpha_rear = np.arctan2(
            -self.v_y + self.rear_wheel_to_center * self.heading_dot,
            max(self.v_x, 1e-3),  # Prevent division by zero
        )
        max_slip_angle = 0.2
        self.alpha_front = np.clip(self.alpha_front, -max_slip_angle, max_slip_angle)
        self.alpha_rear = np.clip(self.alpha_rear, -max_slip_angle, max_slip_angle)

        self.Fy_front = self.tire_stiffness_front * self.alpha_front
        self.Fy_rear = self.tire_stiffness_rear * self.alpha_rear

        self.v_x_dot = (
            self.acceleration
            - (self.Fy_front * np.sin(self.steering)) / self.mass
            + self.v_y * self.heading_dot
        )
        self.v_y_dot = (
            self.Fy_front * np.cos(self.steering) + self.Fy_rear
        ) / self.mass - self.v_x * self.heading_dot
        self.x_dot = self.v_x * np.cos(self.heading) - self.v_y * np.sin(self.heading)
        self.y_dot = self.v_x * np.sin(self.heading) + self.v_y * np.cos(self.heading)
        self.heading_dot_dot = (
            self.Fy_front * self.front_wheel_to_center * np.cos(self.steering)
            - self.Fy_rear * self.rear_wheel_to_center
        ) / self.moment_of_inertia

        # self.velocity += self.acceleration * self.dt
        # self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)

        self.heading_dot += self.heading_dot_dot * self.dt
        self.heading += self.heading_dot * self.dt
        self.heading = self.heading % (2 * np.pi)
        self.v_x += self.v_x_dot * self.dt
        self.v_y += self.v_y_dot * self.dt
        self.velocity = np.sqrt(self.v_x**2 + self.v_y**2)
        self.x += self.x_dot * self.dt
        self.y += self.y_dot * self.dt

    def action_to_input(self, action: dict):
        self.verify_action(action)

        input = np.array(
            [action["throttle"], action["braking"], action["steering_rate"]]
        )

        self.verify_input(input)

        return input

    @staticmethod
    def verify_input(input: np.array) -> np.array:
        input[0] = np.clip(input[0], 0, 1)
        input[1] = np.clip(input[1], 0, 1)

        return input

    @staticmethod
    def verify_action(action: dict) -> dict:
        for key in ["throttle", "braking", "steering_rate"]:
            if key not in action:
                action[key] = 0

        return action

    @staticmethod
    def verify_initial_state(initial_state: dict) -> dict:
        for key in [
            "x",
            "y",
            "heading",
            "steering",
            "slip_angle",
            "velocity",
            "acceleration",
        ]:
            if key not in initial_state or not isinstance(
                initial_state[key], (int, float)
            ):
                initial_state[key] = 0

        return initial_state

    @property
    def state(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "steering": self.steering,
            "slip_angle": self.slip_angle,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
        }
