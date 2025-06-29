from typing import Callable, Dict

enhancement_options: Dict[str, Callable] = {}


def register_enhancement(name: str):
    def decorator(func: Callable):
        enhancement_options[name] = func
        return func

    return decorator
