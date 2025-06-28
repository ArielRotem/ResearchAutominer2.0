
import pandas as pd

def remove_duplicates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame based on specified columns.
    Keeps the first occurrence by default.
    """
    return df.drop_duplicates(subset=columns, keep='first')

def filter_by_value(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """
    Removes rows where the value in the specified column matches the given value.
    """
    return df[df[column] != value]
