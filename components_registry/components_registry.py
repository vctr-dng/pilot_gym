from collections import defaultdict

registry = defaultdict(dict)

DEFAULT_SEPARATOR = "/"
DEFAULT_TYPE = "default"


def get_registry():
    return {k: dict(v) for k, v in registry.items()}


def add_to_registry(type_name, sub_name, cls):
    if sub_name in registry[type_name]:
        raise ValueError(f"'{sub_name}' is already registered under '{type_name}'.")
    # TODO check if provided cls is implementing base class
    registry[type_name][sub_name] = cls


def register(key):
    """
    Decorator to register a class under a specific component type and name.
    """

    def decorator(cls):
        type_name, sep, sub_name = key.partition("/")
        if not sep:
            # TODO add logging
            sub_name = type_name
            type_name = DEFAULT_TYPE
        if not sub_name:
            raise ValueError("The name of the component must be provided.")
        add_to_registry(type_name, sub_name, cls)
        return cls

    return decorator


def get_component(id):
    type_name, sep, sub_name = id.partition("/")
    if not sep:
        sub_name = type_name
        type_name = DEFAULT_TYPE
    if type_name not in registry or sub_name not in registry[type_name]:
        raise ValueError(f"Component '{id}' is not registered.")
    return registry[type_name][sub_name]


def make(id, **kwargs):
    component = get_component(id)
    return component(**kwargs)
