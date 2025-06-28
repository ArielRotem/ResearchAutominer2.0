
import pandas as pd

def remove_duplicates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame based on specified columns.
    Keeps the first occurrence by default.
    """
    return df.drop_duplicates(subset=columns, keep='first')
