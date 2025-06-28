
import pandas as pd

def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    """
    Renames a single column in the DataFrame.
    """
    return df.rename(columns={old_name: new_name})
