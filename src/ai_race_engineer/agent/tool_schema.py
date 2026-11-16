import inspect
from collections.abc import Callable

JSON_TYPES: dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool_schema(tool: Callable[..., object]) -> dict[str, object]:
    """Anthropic tool definition of a typed function: its name, its docstring as the description and every
    argument required. A function with arguments is strict, so the model always sends exactly the declared
    arguments; one without arguments is not, because the API allows only 20 strict tools in a request.
    In:  tool  Callable[..., object]  arguments annotated str, int, float or bool
    Out:       dict[str, object]      {name, description, strict, input_schema}
    """
    parameters = inspect.signature(tool).parameters
    return {
        "name": tool.__name__,
        "description": inspect.getdoc(tool),
        "strict": bool(parameters),
        "input_schema": {
            "type": "object",
            "properties": {name: {"type": JSON_TYPES[p.annotation]} for name, p in parameters.items()},
            "required": list(parameters),
            "additionalProperties": False,
        },
    }
