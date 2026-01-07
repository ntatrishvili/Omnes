from typing import Callable, Any, Dict


class InitOnSet:
    """
    Marker that stores an initializer function and optional default.
    For attributes that are processed whenever the class attribute is assigned.
    """

    def __init__(self, init_fn: Callable[[Any], Any], *, default: Any = None):
        self.init_fn = init_fn
        self.default = default


class InitializingMeta(type):
    """
    Metaclass that intercepts assignments to class attributes listed
    as InitOnSet and replaces assigned values with the return value
    of the corresponding initializer function.
    """

    def __new__(mcls, name, bases, namespace: Dict[str, Any]):
        # Inherit init-map from bases
        init_map: Dict[str, Callable[[Any], Any]] = {}
        for b in bases:
            init_map.update(getattr(b, "_init_on_set", {}))
        # Collect InitOnSet in this class namespace and set initial defaults
        for k, v in list(namespace.items()):
            if isinstance(v, InitOnSet):
                init_map[k] = v.init_fn
                namespace[k] = v.init_fn(v.default)
        cls = super().__new__(mcls, name, bases, namespace)
        cls._init_on_set = init_map
        return cls

    def __setattr__(cls, name: str, value: Any):
        init_map = getattr(cls, "_init_on_set", {})
        if name in init_map:
            value = init_map[name](value)
        super().__setattr__(name, value)
