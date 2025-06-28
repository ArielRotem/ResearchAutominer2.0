
import pandas as pd
from backend.functions.utilities import column_name_to_index

def add_row_index_column(data, col_name="Index", first_position=True):
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

def is_empty(data, column_name, new_column_name, value_empty=1, value_not_empty=0 ):
    """
    Checks if each cell in the specified column is not empty.
    
    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    column_index = column_name_to_index(data, column_name)
    data[new_column_name] = data.iloc[:, column_index].apply(lambda x: value_not_empty if pd.notna(x) and x != '' else value_empty)
    return data

def create_indicator_column_by_keyword(data, source_column, keyword, new_column):
    """
    Creates a new column with 1 if the source column contains the keyword, else 0.
    """
    data[new_column] = data[source_column].apply(
        lambda x: 1 if isinstance(x, str) and keyword in x else 0
    )
    return data

def create_column_from_value_map(data, source_column, new_column, value_map, default_value=""):
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

def split_gestational_age(data, column_name='gestational age', week_col='gestational_week', day_col='gestational_day'):
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
            # Get the digits after the decimal point, then round just in case of 38.1999
            return int(round((float(x) - int(float(x))) * 10))
        except:
            return ""

    data[week_col] = data.iloc[:, column_index].apply(extract_week)
    data[day_col] = data.iloc[:, column_index].apply(extract_day)

    return data

def sum_two_columns_threshold(data, col1_name, col2_name, new_column_name, threshold, above_value=1, below_value=0, default_value=""):
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
