from components_registry import register
from dynamics_model import DynamicModel


@register("state_observer/base_state_observer")
class BaseStateObserver:
    def __init__(self, dynamic_model: DynamicModel, observed_state: list):
        self.dynamic_model = dynamic_model
        self.observed_state = observed_state
        self.observation_size = len(observed_state)

    def observe(self) -> dict:
        observation = dict()

        for state in self.observed_state:
            try:
                observation[state] = self.dynamic_model[state]
            except (KeyError, TypeError):
                try:
                    observation[state] = getattr(self.dynamic_model, state)
                except AttributeError:
                    raise Exception(
                        f"State '{state}' not found in dynamic model\
                        {self.dynamic_model.__class__}"
                    )

        return observation
