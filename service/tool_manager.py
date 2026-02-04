import inspect
from typing import Callable, Any


class ToolManager:
    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}

    def register_tool(self, func: Callable) -> None:
        name = func.__name__
        self._tools[name] = func
        self._schemas[name] = self._build_schema(func)

    def register_switch_tool(self) -> None:
        def switch_to_reasoning(reason: str) -> dict:
            """복잡한 추론이 필요할 때 호출합니다."""
            return {"switch": True, "reason": reason}

        self.register_tool(switch_to_reasoning)

    def _build_schema(self, func: Callable) -> dict:
        sig = inspect.signature(func)
        doc = func.__doc__ or ""

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in (int, float):
                    param_type = "integer" if param.annotation == int else "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"

            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}",
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            else:
                properties[param_name]["default"] = param.default

        return {
            "name": func.__name__,
            "description": doc.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def get_tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_tool_schemas(self) -> list[dict]:
        return list(self._schemas.values())

    def get_tools(self) -> list[Callable]:
        return list(self._tools.values())

    def execute_tool(self, name: str, args: dict) -> Any:
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name](**args)
