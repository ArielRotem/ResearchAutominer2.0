
import pandas as pd
from typing import List, Dict, Any
from backend.core.function_loader import load_functions

def run_manuscript(df: pd.DataFrame, manuscript: List[Dict[str, Any]]) -> pd.DataFrame:
    """Executes a manuscript (a list of functions and their arguments) on a DataFrame."""
    functions = load_functions()
    
    for step in manuscript:
        func_name = step.get("name")
        params = step.get("params", {})
        
        if func_name not in functions:
            raise ValueError(f"Function '{func_name}' not found.")
            
        func_data = functions[func_name]
        func = func_data["callable"]
        
        # Prepare the arguments for the function call
        # This is a simplified version. A real implementation would be more robust.
        kwargs = {k: v for k, v in params.items()}
        
        # The first argument to all our functions is the DataFrame
        df = func(df, **kwargs)
        
    return df
