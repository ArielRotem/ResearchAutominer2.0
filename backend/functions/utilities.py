
import pandas as pd

def excel_column_to_index(col):
    """
    Convert an Excel-style column label to a zero-based column index.
    Example: 'A' -> 0, 'B' -> 1, 'Z' -> 25, 'AA' -> 26, 'BC' -> 54
    """
    index = 0
    for c in col:
        index = index * 26 + (ord(c.upper()) - ord('A') + 1)
    return index - 1  # convert to zero-based index

def column_name_to_index(data, column_name):
    """
    Given a DataFrame and a column name, return the zero-based index of the column.
    Raises a KeyError if the column is not found.
    """
    if column_name in data.columns:
        return data.columns.get_loc(column_name)
    else:
        raise KeyError(f"Column '{column_name}' not found in the DataFrame.")
