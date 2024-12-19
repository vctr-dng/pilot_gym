import numpy as np

from components_registry import register
from dynamics_model import DynamicModel


@register("dynamic_model/kinematic_bicycle")
class KinematicBicycleModel(DynamicModel):
    actions = [
        "throttle",
        "braking",
        "steering",
    ]

    def __init__(self, vehicle_params: dict, simulation_params: dict):
        self.x = 0  # m
        self.y = 0  # m
        self.heading = 0  # radian
        self.steering = 0  # radian
        self.slip_angle = 0  # radian
        self.velocity = 0  # m/s
        self.acceleration = 0  # m/s^2

        self.wheelbase = vehicle_params["wheelbase"]
        self.rear_wheel_to_center = vehicle_params["rear_wheel_to_center"]
        self.front_wheel_to_center = self.wheelbase - self.rear_wheel_to_center
        self.max_acceleration = vehicle_params["max_acceleration"]
        self.min_acceleration = np.abs(vehicle_params["min_acceleration"])
        self.max_velocity = vehicle_params["max_velocity"]
        self.lock_to_lock_steering = vehicle_params["lock_to_lock_steering"]

        self.dt = simulation_params["dt"]
        
        self.reset_actions()
    
    def reset_actions(self):
        for action in self.actions:
            self.__setattr__(action, 0)
            self.__setattr__(f"previous_{action}", 0)
    
    def update_actions(self, actions: dict):
        for action in self.actions:
            self.__setattr__(action, actions[action])
            if action == 'throttle':
                self.throttle *= self.max_acceleration
            elif action == 'braking':
                self.braking *= self.min_acceleration
            elif action == 'steering':
                self.steering *= self.lock_to_lock_steering / 2
                self.steering = np.clip(
                    self.steering,
                    -self.lock_to_lock_steering / 2,
                    self.lock_to_lock_steering / 2,
                )
    
    def update_previous_actions(self):
        for action in self.actions:
            self.__setattr__(f"previous_{action}", self.__getattribute__(action))

    def reset(self, initial_state: dict):
        self.verify_initial_state(initial_state)

        self.x = initial_state["x"]
        self.y = initial_state["y"]
        self.heading = initial_state["heading"]
        self.steering = initial_state["steering"]
        self.slip_angle = initial_state["slip_angle"]
        self.velocity = initial_state["velocity"]
        self.acceleration = initial_state["acceleration"]
        
        self.reset_actions()

    def step(self, action: dict):
        self.verify_action(action)
        self.update_previous_actions()
        self.update_actions(action)

        self.acceleration = (
            self.throttle - self.braking
        )
        self.acceleration = np.clip(self.acceleration, -self.min_acceleration, self.max_acceleration)

        self.slip_angle = np.arctan(
            self.rear_wheel_to_center * np.tan(self.steering) / self.wheelbase
        )

        self.velocity += self.acceleration * self.dt
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)

        heading_dot = (
            self.velocity * np.sin(self.slip_angle) / self.rear_wheel_to_center
        )
        self.heading += heading_dot * self.dt

        x_dot = self.velocity * np.cos(self.heading + self.slip_angle)
        y_dot = self.velocity * np.sin(self.heading + self.slip_angle)
        self.x += x_dot * self.dt
        self.y += y_dot * self.dt

    def action_to_input(self, action: dict):
        self.verify_action(action)
        # input = np.array(
        #     [action["throttle"], action["braking"], action["steering_rate"]]
        # )
        input = np.array(
            [value for key, value in action.items()]
        )
        self.verify_input(input)
        return input

    @staticmethod
    def verify_input(input: np.array) -> np.array:
        input[0] = np.clip(input[0], 0, 1)
        input[1] = np.clip(input[1], 0, 1)

        return input

    @classmethod
    def verify_action(cls, action: dict) -> dict:
        for key in cls.actions:
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
            if key not in initial_state:
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
