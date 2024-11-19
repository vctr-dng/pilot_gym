from gymnasium import make, register

from .dynamic_model import DynamicModel as DynamicModel
from .bicycle_model import BicycleModel as BicycleModel

# TODO custom make register logic


def register_dynamic_model(id: str, dynamic_model: DynamicModel):
    register(id=f"dynamic_model/{id}", entry_point=dynamic_model)


def make_dynamic_model(id: str, **kwargs):
    return make(id=f"dynamic_model/{id}", **kwargs)
