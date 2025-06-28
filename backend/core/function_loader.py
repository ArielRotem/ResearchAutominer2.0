
import os
import importlib
import inspect
from typing import Callable, Dict, Any

FUNCTIONS_DIR = "backend/functions"

def load_functions() -> Dict[str, Callable]:
    """Dynamically loads all functions from the functions directory and its submodules."""
    functions = {}
    for root, _, files in os.walk(FUNCTIONS_DIR):
        for filename in files:
            if filename.endswith(".py") and not filename.startswith("__"):
                # Construct module path relative to the project root
                relative_path = os.path.relpath(os.path.join(root, filename), start=os.getcwd())
                module_name = relative_path.replace(os.sep, '.')[:-3]
                
                try:
                    module = importlib.import_module(module_name)
                    for name, func in inspect.getmembers(module, inspect.isfunction):
                        # Ensure it's a function defined in this module, not an imported one
                        if func.__module__ == module_name:
                            functions[name] = {
                                "callable": func,
                                "name": name,
                                "doc": inspect.getdoc(func),
                                "params": list(inspect.signature(func).parameters.keys())
                            }
                except Exception as e:
                    print(f"Error loading module {module_name}: {e}")
    return functions
