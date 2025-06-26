from typing import Callable


def get_enhance_fn(name: str) -> Callable:
    if name == "passthrough":
        from inference.passthrough import process_session

        return process_session
    elif name == "baseline":
        return print

    raise ValueError(f"Enhance option {name} not recognised. Add code here!")
