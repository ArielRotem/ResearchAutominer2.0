import pandas as pd
from typing import Any
from backend.functions.utilities import column_name_to_index

def add_row_index_column(data: pd.DataFrame, col_name: str = "Index", first_position: bool = True) -> pd.DataFrame:
    """
    Adds a 1-based row index column to the DataFrame.
    If first_position is True, inserts it as the first column.
    """
    indexed = data.copy()
    indexed[col_name] = range(1, len(indexed) + 1)
    
    if first_position:
        cols = [col_name] + [col for col in indexed.columns if col != col_name]
        indexed = indexed[cols]
    
    return indexed

def is_empty(data: pd.DataFrame, column_name: str, new_column_name: str, value_empty: Any = 1, value_not_empty: Any = 0 ) -> pd.DataFrame:
    """
    Checks if each cell in the specified column is not empty.
    
    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    column_index = column_name_to_index(data, column_name)
    data[new_column_name] = data.iloc[:, column_index].apply(lambda x: value_not_empty if pd.notna(x) and str(x).strip() != '' else value_empty)
    return data

def create_indicator_column_by_keyword(data: pd.DataFrame, source_column: str, keyword: str, new_column: str) -> pd.DataFrame:
    """
    Creates a new column with 1 if the source column contains the keyword, else 0.
    """
    data[new_column] = data[source_column].apply(
        lambda x: 1 if isinstance(x, str) and keyword in x else 0
    )
    return data

def create_column_from_value_map(data: pd.DataFrame, source_column: str, new_column: str, value_map: dict, default_value: Any = "") -> pd.DataFrame:
    """
    Creates a new column based on mapping specific values from an existing column.
    """
    col_index = column_name_to_index(data, source_column)

    def map_value(x):
        try:
            x_numeric = int(float(x))
            return value_map.get(x_numeric, default_value)
        except:
            return default_value

    data[new_column] = data.iloc[:, col_index].apply(map_value)
    return data

def split_gestational_age(data: pd.DataFrame, column_name: str = 'gestational age', week_col: str = 'gestational_week', day_col: str = 'gestational_day') -> pd.DataFrame:
    """
    Splits a gestational age column in dot-decimal format (e.g., 38.5 means 38 weeks + 5 days)
    into two new columns: gestational_week and gestational_day.
    """
    column_index = column_name_to_index(data, column_name)

    def extract_week(x):
        try:
            return int(float(x))
        except:
            return ""

    def extract_day(x):
        try:
            return int(round((float(x) - int(float(x))) * 10))
        except:
            return ""

    data[week_col] = data.iloc[:, column_index].apply(extract_week)
    data[day_col] = data.iloc[:, column_index].apply(extract_day)

    return data

def sum_two_columns_threshold(data: pd.DataFrame, col1_name: str, col2_name: str, new_column_name: str, threshold: float, above_value: Any = 1, below_value: Any = 0, default_value: Any = "") -> pd.DataFrame:
    """
    Creates a new column where the value is:
    - above_value if col1 + col2 >= threshold
    - below_value if col1 + col2 < threshold
    - default_value if either is missing or non-numeric
    """
    idx1 = column_name_to_index(data, col1_name)
    idx2 = column_name_to_index(data, col2_name)

    def compute(row):
        try:
            val1 = float(row.iloc[idx1])
            val2 = float(row.iloc[idx2])
            return above_value if (val1 + val2) >= threshold else below_value
        except:
            return default_value

    data[new_column_name] = data.apply(compute, axis=1)
    return data

def flag_if_column_contains_any_value(data: pd.DataFrame, column_name: str, target_values: list[Any], result_col: str) -> pd.DataFrame:
    """
    Creates a new column, flagging 1 if the source column contains any of the target values, else 0.
    """
    data[result_col] = data[column_name].apply(
        lambda x: 1 if any(str(val) in str(x) for val in target_values) else 0
    )
    return data

def columns_contain_nonzero_nonfalse(data: pd.DataFrame, column_names: list[str], result_col: str) -> pd.DataFrame:
    """
    Creates a new column, returning 1 if any of the specified columns contain a non-zero, non-false value, else 0.
    """
    def check_row(row):
        for col_name in column_names:
            val = row[col_name]
            if pd.notna(val) and val != 0 and val != False and val != "0" and val != "False" and val != "":
                return 1
        return 0
    data[result_col] = data.apply(check_row, axis=1)
    return data
