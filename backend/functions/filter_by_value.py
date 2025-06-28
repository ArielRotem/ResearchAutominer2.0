
import pandas as pd

def filter_by_value(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """
    Removes rows where the value in the specified column matches the given value.
    """
    return df[df[column] != value]
