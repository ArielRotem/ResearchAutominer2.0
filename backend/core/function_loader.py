
import os
import importlib
import inspect
from typing import Callable, Dict, Any

FUNCTIONS_DIR = "backend/functions"

def load_functions() -> Dict[str, Callable]:
    """Dynamically loads all functions from the functions directory."""
    functions = {}
    for filename in os.listdir(FUNCTIONS_DIR):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = f"backend.functions.{filename[:-3]}"
            module = importlib.import_module(module_name)
            for name, func in inspect.getmembers(module, inspect.isfunction):
                if func.__module__ == module_name:
                    functions[name] = {
                        "callable": func,
                        "name": name,
                        "doc": inspect.getdoc(func),
                        "params": inspect.signature(func).parameters
                    }
    return functions
