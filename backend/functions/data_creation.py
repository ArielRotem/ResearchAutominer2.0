def flag_if_column_contains_any_value(data, column_name, target_values, result_col):
    """
    Creates a new column, flagging 1 if the source column contains any of the target values, else 0.
    """
    data[result_col] = data[column_name].apply(
        lambda x: 1 if any(str(val) in str(x) for val in target_values) else 0
    )
    return data

def columns_contain_nonzero_nonfalse(data, column_names, result_col):
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