from components_registry import register
from dynamics_model import DynamicModel


@register("state_observer/base_state_observer")
class BaseStateObserver:
    def __init__(self, dynamic_model: DynamicModel, observed_state: list):
        self.dynamic_model = dynamic_model
        self.observed_state = observed_state
        self.observation_size = len(observed_state)

    def __call__(self) -> dict:
        observation = dict()

        for state_name in self.observed_state.keys():
            observation[state_name] = self.query(state_name)

        return observation

    def query(self, state_name: str) -> float:
        state_value: float = None

        if state_name not in self.observed_state.keys():
            raise Exception(f"State '{state_name}' not found in observed state")

        info = self.observed_state[state_name]

        try:
            state_value = self.dynamic_model[info["path"]]
        except (KeyError, TypeError):
            try:
                state_value = getattr(self.dynamic_model, info["path"])
            except AttributeError:
                raise Exception(
                    f"State '{state_name}' not found in dynamic model\
                    {self.dynamic_model.__class__}"
                )

        if state_value is None:
            raise Exception(f"Could not query state '{state_name}'")

        return state_value
