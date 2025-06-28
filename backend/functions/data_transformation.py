

import pandas as pd
import re
from backend.functions.utilities import column_name_to_index

def update_dataframe(originalData, col1, words1, logical_op, col2, words2, col3, step_size, num_steps, col3_empty, result_column_name, return_values=True, unique=False, dictionary=None, limitResults=None):    
    data = originalData.copy()
    # Prepare column indices
    col1_index = column_name_to_index(data, col1) if col1 else None
    col2_index = column_name_to_index(data, col2) if col2 else None
    col3_index = column_name_to_index(data, col3) if col3 else None

    # Function to apply on each row
    def evaluate_row(row):
        concatenated_results = []  # List to collect valid results

        for step in range(num_steps):
            offset = step * step_size
            current_col1_index = col1_index + offset if col1_index is not None else None
            current_col2_index = col2_index + offset if col2_index is not None else None
            current_col3_index = col3_index + offset if col3_index is not None else None

            # Ensure column indexes are within the bounds of the row
            if (current_col1_index and current_col1_index >= len(row)) or (current_col2_index and current_col2_index >= len(row)) or (current_col3_index and current_col3_index >= len(row)):
                continue

            # Convert the data to string for regex matching
            data_col1 = str(row.iloc[current_col1_index]) if current_col1_index is not None else ''
            data_col2 = str(row.iloc[current_col2_index]) if current_col2_index is not None else ''
            data_col3 = str(row.iloc[current_col3_index]) if current_col3_index is not None else ''
            
            # Check conditions
            cond1 = any(word in data_col1 for word in words1) if col1 and words1 else True
            cond2 = any(word in data_col2 for word in words2) if words2 else True
            cond3 = True if col3_index is None else (bool(data_col3) if not col3_empty else not data_col3)
            
            # Adjust logic operation based on whether col1 is effectively in use
            if col1 and words1:
                condition = (cond1 and cond2) if logical_op == 'AND' else (cond1 or cond2)
            else:
                condition = cond1 and cond2

            # Collect results if conditions are met
            if condition and cond3:
                if return_values:
                    concatenated_results.extend([item for item in data_col3.split(';') if item.strip()])

                else:
                    concatenated_results.append(True)  # Collect 'True' values for existence check
                    break # one true is enough

        # Final assignment based on operation mode
        if return_values:
            if dictionary:
                # Replace values based on dictionary with default value "Other" if key not found
                concatenated_results = [dictionary.get(item, "Uncategorized") for item in concatenated_results]

            if unique:
                # Deduplicate by converting to set and back to list
                concatenated_results = list(set(concatenated_results))
            
            if limitResults is not None and len(concatenated_results) > limitResults:
                # Handle the limitResults parameter
                concatenated_results = concatenated_results[:limitResults]
            
            concatenated_results = sorted(concatenated_results, key=lambda x: ('' if x == 'Uncategorized' else x))
            return ', '.join(map(str, concatenated_results)) if concatenated_results else ''

        else:
            return int(any(concatenated_results)) ## swap from true to: true=1, false=0


    # Apply the function and assign results to the new result column
    results = data.apply(evaluate_row, axis=1)
    data.loc[:, result_column_name] = results
    return data

def containswords_andor_containswords_and_nonempty_result_values(data, col1, words1, logical_op, col2, words2, col3, step_size, num_steps, result_column_name, unique=True, dictionary=None, limitResults=None):
    """
    Alias for update_dataframe: checks for words in two columns with logical AND/OR, and extracts non-empty result values.
    """
    return update_dataframe(data, col1, words1, logical_op, col2, words2, col3, step_size, num_steps, False, result_column_name, return_values=True, unique=unique, dictionary=dictionary, limitResults=limitResults)

def containswords_and_nonempty_result_values(data, col2, words2, col3, step_size, num_steps, result_column_name, unique=True, dictionary=None, limitResults=None):
    """
    Alias for update_dataframe: checks for words in one column, and extracts non-empty result values.
    """
    return update_dataframe(data, '', '', '', col2, words2, col3, step_size, num_steps, False, result_column_name, return_values=True, unique=unique, dictionary=dictionary, limitResults=limitResults)

def containswords_andor_containswords_and_nonempty_result_exists(data, col1, words1, logical_op, col2, words2, col3, step_size, num_steps, result_column_name):
    """
    Alias for update_dataframe: checks for words in two columns with logical AND/OR, and checks if non-empty result exists.
    """
    return update_dataframe(data, col1, words1, logical_op, col2, words2, col3, step_size, num_steps, False, result_column_name, return_values=False, unique=False, dictionary=None)

def containswords_andor_containswords_result_exists(data, col1, words1, logical_op, col2, words2, step_size, num_steps, result_column_name):
    """
    Alias for update_dataframe: checks for words in two columns with logical AND/OR, and checks if result exists.
    """
    return update_dataframe(data, col1, words1, logical_op, col2, words2, '', step_size, num_steps, False, result_column_name, return_values=False, unique=False, dictionary=None)

def containswords_result_exists(data, col2, words2, step_size, num_steps, result_column_name):
    """
    Alias for update_dataframe: checks for words in one column, and checks if result exists.
    """
    return update_dataframe(data, '', '', '', col2, words2, '', step_size, num_steps, True, result_column_name, return_values=False, unique=False, dictionary=None)

def remove_rows_below_threshold(data, column_name, threshold):
    """
    Removes rows where the value in the specified column is below the given threshold.
    """
    return data[data.iloc[:, column_name_to_index(data, column_name)] >= threshold]
    
def remove_rows_above_threshold(data, column_name, threshold):
    """
    Removes rows where the value in the specified column is above the given threshold.
    """
    return data[data.iloc[:, column_name_to_index(data, column_name)] <= threshold]

def remove_rows_if_contains(data, column_name, words):
    """
    Removes rows if the specified column contains any of the given words (case-insensitive).
    """
    return data[~data.iloc[:, column_name_to_index(data, column_name)].astype(str).str.lower().apply(lambda x: any(word.lower() in x for word in words))]

def update_column_with_values(data, column_name, words_dict, default_value="0", empty_value=None):
    """
    Updates values in a column based on a dictionary of words. If a cell contains any word from a key's list, it gets the key's value.
    """
    column_index = column_name_to_index(data, column_name)

    def replace_value(cell_value):
        cell_value_lower = cell_value.lower()  # Lowercase the cell value for case-insensitive comparison
        if cell_value == "":  # Check if the cell is empty
            return empty_value if empty_value is not None else cell_value
        for key, words in words_dict.items():
            if any(word.lower() in cell_value_lower for word in words):
                return key
        if cell_value != "":
            return default_value  # Use the specified default value if no matches found
        return ""

    data.iloc[:, column_index] = data.iloc[:, column_index].astype(str).apply(replace_value)
    return data

def clear_negative_values(data, column_name):
    """
    Clears negative numeric values in a specified column by replacing them with an empty string.
    """
    col_index = column_name_to_index(data, column_name)
    temp_series = pd.to_numeric(data.iloc[:, col_index], errors='coerce')
    negative_mask = temp_series < 0
    data.loc[negative_mask, data.columns[col_index]] = ''
    return data

def clear_strings_multiple_columns(data, column_names, words, indicator=-1):
    """
    Clears strings in multiple columns based on a list of words. Can also clear an adjacent column.
    """
    words_lower = [word.lower() for word in words]
    
    for column_name in column_names:
        col_index = column_name_to_index(data, column_name)
        adj_col_index = col_index + indicator if indicator != 0 else col_index
        
        if not (0 <= adj_col_index < data.shape[1]):
            continue
        
        mask = data.iloc[:, col_index].astype(str).str.lower().apply(lambda x: any(word in x for word in words_lower))
        
        data.iloc[mask, col_index] = ""
        if indicator != 0:
            data.iloc[mask, adj_col_index] = ""
    
    return data

def custom_logic_operation(data, col1_name, col2_name, result_col_name):
    """
    Applies custom logic to two columns to create a new result column based on specific values (1, 0, 2).
    """
    data_copy = data.copy()
    col1_idx = column_name_to_index(data_copy, col1_name)
    col2_idx = column_name_to_index(data_copy, col2_name)
    
    def apply_logic(row):
        col1 = row[col1_idx]
        col2 = row[col2_idx]
        
        if pd.isna(col1) and pd.isna(col2):
            return None
        
        if col1 == "1" or col2 == "1":
            return 1
        if col1 == "0" or col2 == "0":
            return 0
        if col1 == "2" or col2 == "2":
            return 2
        
        return None

    data_copy[result_col_name] = data.apply(apply_logic, axis=1)
    return data_copy

def concat_unique_values(data, column_names, new_column_name, limitResults=None):
    """
    Concatenates unique values from specified columns into a new column, with an optional limit.
    """
    data_copy = data.copy()
    column_indices = [column_name_to_index(data_copy, name) for name in column_names]
    
    def process_row(row):
        values = [row.iloc[idx] for idx in column_indices if pd.notna(row.iloc[idx]) and row.iloc[idx] != '']
        unique_values = list(set(values))
        
        if limitResults is not None:
            unique_values = unique_values[:limitResults]
        
        sorted_values = sorted(unique_values)
        return ', '.join(sorted_values)
    
    data_copy[new_column_name] = data_copy.apply(process_row, axis=1)
    return data_copy

def remove_columns(data, column_names):
    """
    Removes specified columns from a DataFrame based on column names or ranges.
    """
    columns_to_remove = []
    for col in column_names:
        if '~' in col:
            start_col, end_col = col.split('~')
            start_index = column_name_to_index(data, start_col)
            end_index = column_name_to_index(data, end_col)
            columns_to_remove.extend(data.columns[start_index:end_index+1])
        else:
            index = column_name_to_index(data, col)
            columns_to_remove.append(data.columns[index])

    data.drop(columns=columns_to_remove, inplace=True)
    return data

def clear_values_based_on_reference(data, target_column_name, reference_column_name, reference_value):
    """
    Clears values in the target column based on a condition met in the reference column.
    """
    target_index = column_name_to_index(data, target_column_name)
    reference_index = column_name_to_index(data, reference_column_name)

    def clear_value(row):
        if row.iloc[reference_index] == reference_value:
            return ""
        else:
            return row.iloc[target_index]

    data.iloc[:, target_index] = data.apply(clear_value, axis=1)
    return data

def filter_numbers(data, column_name, lowerThan=None, higherThan=None, emptyOk=True):
    """
    Filters numbers in a column based on specified thresholds, replacing values outside the range with empty strings.
    """
    column_index = column_name_to_index(data, column_name)
    def filter_val(x):
        try:
            if (emptyOk and (x is None or x == "")):
                return x
            num = float(x)
            if (lowerThan is not None and num < lowerThan) or (higherThan is not None and num > higherThan):
                return ''
            else:
                return x
        except ValueError:
            return x
    data.iloc[:, column_index] = data.iloc[:, column_index].apply(filter_val)
    return data

def flip_sign(data, column_name):
    """
    Flips the sign of numeric values in the specified column.
    """
    col_index = column_name_to_index(data, column_name)
    def flip(x):
        try:
            return -float(x)
        except ValueError:
            return x
    data.iloc[:, col_index] = data.iloc[:, col_index].apply(flip)
    return data
    
def multiply_by_number(data, column_name, multiplier):
    """
    Multiplies the values in the specified column by the multiplier.
    """
    column_index = column_name_to_index(data, column_name)
    def multiply(x):
        try:
            return float(x)*multiplier
        except ValueError:
            return x
    data.iloc[:, column_index] = data.iloc[:, column_index].apply(multiply)
    return data    

def cutoff_number(data, column_name, new_column_name, cutoff, above=1, below=0, empty_value=None):
    """
    Creates a new column indicating if a numeric value is above or below a cutoff.
    """
    column_index = column_name_to_index(data, column_name)
    def check_cutoff(x):
        try:
            if float(x) >= cutoff:
                return above
            else:
                return below
        except ValueError:
            return empty_value

    data[new_column_name] = data.iloc[:, column_index].apply(check_cutoff)
    return data

def compare_values(data, column_name, new_column_name, target_value, match_return, no_match_return):
    """
    Compares values in a column to a target value, returning match_return or no_match_return.
    """
    column_index = column_name_to_index(data, column_name)
    
    def check_match(x):
        try:
            if float(x) == float(target_value):
                return match_return
        except ValueError:
            if str(x) == str(target_value):
                return match_return
        return no_match_return

    data[new_column_name] = data.iloc[:, column_index].apply(check_match)
    return data

def combine_columns(data, column_names, new_column_name, delimiter=" "):
    """
    Combines multiple columns into a new column, removing duplicates and empty values.
    """
    column_indices = [column_name_to_index(data, name) for name in column_names]
    
    def combine_row(row):
        values = [str(row.iloc[idx]) for idx in column_indices]
        filtered_values = [value for value in values if value and value != 'nan']
        filtered_values = list(set(filtered_values))
        
        if '0' in filtered_values and len(filtered_values) > 1:
            filtered_values.remove('0')
        
        return delimiter.join(filtered_values)

    data[new_column_name] = data.apply(combine_row, axis=1)
    return data

def remove_contaminant_and_count(data, column_name, new_column_name, delimiter=',', default_value='', contaminant=''):
    """
    Processes a column by splitting, removing a contaminant, and returning a count or default value.
    """
    def process_cell(cell):
        if pd.isna(cell) or cell == '':
            return default_value
        
        elements = cell.split(delimiter)
        elements = [elem.strip() for elem in elements if elem.strip() != '' and elem.strip() != contaminant]
        
        num_elements = len(elements)
        
        if num_elements == 0:
            return '0'
        elif num_elements == 1:
            return '1'
        else:
            return '2'
    
    data[new_column_name] = data[column_name].apply(process_cell)
    return data

def does_column_contain_string_in_category_list(data, column_name, new_column_name, search_list, delimiter=',', empty_value=''):
    """
    Checks if a column cell contains any string from a search list, returning 1 for match, 0 for no match.
    """
    def process_cell(cell):
        if pd.isna(cell) or cell == '':
            return empty_value
        
        elements = cell.split(delimiter)
        
        if any(search_string in [elem.strip() for elem in elements] for search_string in search_list):
            return 1
        else:
            return 0
    
    data[new_column_name] = data[column_name].apply(process_cell)
    return data

def process_column_tuples(data, start_column, columns, num_tuples, transformations=None, default_value=None, delimiter=" - "):
    """
    Processes groups of 3 columns, creating new columns based on transformations.
    """
    data_copy = data.copy()
    start_index = column_name_to_index(data_copy, start_column) if isinstance(start_column, str) else start_column

    for i in range(num_tuples):
        col1_index = start_index + i * columns
        col2_index = col1_index + 1
        col3_index = col1_index + 2

        if col3_index >= len(data_copy.columns):
            break

        col1_name = data_copy.columns[col1_index]
        col2_name = data_copy.columns[col2_index]
        col3_name = data_copy.columns[col3_index]

        def process_row(row):
            col1_value = row.iloc[col1_index]
            col2_value = row.iloc[col2_index]
            col3_value = row.iloc[col3_index]

            if pd.notna(col1_value) and pd.notna(col2_value):
                combined_name = f"{col1_value}{delimiter}{col2_value}"
            elif pd.notna(col1_value) or pd.notna(col2_value):
                combined_name = col1_value if pd.notna(col1_value) else col2_value
            else:
                return None

            if transformations:
                transformed_value = transformations.get(
                    str(col3_value),
                    col3_value if default_value is None else default_value
                )
            else:
                transformed_value = col3_value

            return combined_name, transformed_value

        new_column_data = data_copy.apply(process_row, axis=1)

        new_column_names = [item[0] if item else None for item in new_column_data]
        new_column_values = [item[1] if item else None for item in new_column_data]

        for idx, (name, value) in enumerate(zip(new_column_names, new_column_values)):
            if name:
                data_copy.at[idx, name] = value

    return data_copy

def concat_values_across_batches(data, nth_column, step_size, num_batches, output_column_name):
    """
    Concatenates values from specific columns across batches, removing duplicates and empty values.
    """
    start_index = column_name_to_index(data, nth_column)
    output_values = []

    for idx, row in data.iterrows():
        value_set = set()

        for step in range(num_batches):
            column_index = start_index + step * step_size
            if column_index >= len(data.columns):
                break

            value = row.iloc[column_index]
            if pd.notna(value) and value != "":
                value_set.add(value)

        output_values.append(", ".join(sorted(value_set)))

    data[output_column_name] = output_values
    return data

def extract_and_filter_raw_map(data, input_column, substrings, new_column_name):
    """
    Extracts a dictionary from raw map-like data, filters by substrings, and saves to a new column.
    """
    substrings_lower = [s.lower() for s in substrings]

    def process_row(row):
        tokens = row.split(";")
        row_dict = {}
        duplicate_keys = set()

        for token in tokens:
            if "Key:" in token and "Value:" in token:
                key_value = token.split("Value:", 1)
                key = key_value[0].replace("Key:", "").strip()
                value = key_value[1].strip() if len(key_value) > 1 else ""
            elif "Key:" in token:
                key = token.replace("Key:", "").strip()
                value = ""
            else:
                continue

            if key in row_dict:
                duplicate_keys.add(key)
            row_dict[key] = value

        filtered_dict = {
            k: v for k, v in row_dict.items() if any(sub in k.lower() for sub in substrings_lower)
        }
        return filtered_dict

    data[new_column_name] = data[input_column].apply(process_row)
    return data

def categorize_packed_cells(data, before_col, after_col, step, num_before_batches, num_after_batches, result_received, result_before, result_after):
    """
    Categorizes whether packed cells were received before/after birth and counts occurrences.
    """
    before_idx = column_name_to_index(data, before_col)
    after_idx = column_name_to_index(data, after_col)

    def process_row(row):
        count_before = 0
        count_after = 0

        for i in range(num_before_batches):
            idx = before_idx + (i * step)
            if pd.notna(row.iloc[idx]) and str(row.iloc[idx]).strip() != "":
                count_before += 1

        for i in range(num_after_batches):
            idx = after_idx + (i * step)
            if pd.notna(row.iloc[idx]) and str(row.iloc[idx]).strip() != "":
                count_after += 1

        received = 1 if (count_before + count_after) > 0 else 0
        return pd.Series([received, count_before, count_after])

    data[[result_received, result_before, result_after]] = data.apply(process_row, axis=1)
    return data

def categorize_uterotonics(data, base_col, step, num_batches, result_col, cytotec_words, methergin_words):
    """
    Categorizes uterotonic administration (Cytotec, Methergin, both, or none).
    """
    base_idx = column_name_to_index(data, base_col)
   
    def process_row(row):
        cytotec_detected = False
        methergin_detected = False
       
        for batch in range(num_batches):
            idx = base_idx + (batch * step)
            if pd.notna(row.iloc[idx]):
                value_lower = str(row.iloc[idx]).lower()
                if any(word.lower() in value_lower for word in cytotec_words):
                    cytotec_detected = True
                if any(word.lower() in value_lower for word in methergin_words):
                    methergin_detected = True
       
        if cytotec_detected and methergin_detected:
            return 3
        elif cytotec_detected:
            return 1
        elif methergin_detected:
            return 2
        return 0
   
    data[result_col] = data.apply(process_row, axis=1)
    return data

def categorize_full_dilation(data, column, result_col):
    """
    Categorizes a column as 1 if value is 10 (full dilation), else 0.
    """
    col_idx = column_name_to_index(data, column)

    def is_full_dilation(row):
        value = row.iloc[col_idx]
        if pd.isna(value):
            return ""
        try:
            return 1 if float(value) == 10 else 0
        except:
            return ""

    data[result_col] = data.apply(is_full_dilation, axis=1)
    return data

def categorize_surgery_time(data, columns, result_col_label, result_col_numeric):
    """
    Categorizes surgery time into Day, Evening, or Night based on time of day.
    """
    col_indices = [column_name_to_index(data, col) for col in columns]

    def classify_time(row):
        for idx in col_indices:
            cell = row.iloc[idx]
            if pd.isna(cell) or str(cell).strip() == "":
                continue
            try:
                hour = pd.to_datetime(cell).hour
                if 7 <= hour < 16:
                    return ("Day", 0)
                elif 16 <= hour < 21:
                    return ("Evening", 1)
                else:
                    return ("Night", 2)
            except Exception:
                continue
        return ("", "")

    results = data.apply(classify_time, axis=1)
    data[result_col_label] = results.apply(lambda x: x[0])
    data[result_col_numeric] = results.apply(lambda x: x[1])
    return data

def calculate_duration(data, start_column, end_column, result_column):
    """
    Calculates duration in hours (2 decimal places) between two datetime columns.
    """
    start_idx = column_name_to_index(data, start_column)
    end_idx = column_name_to_index(data, end_column)

    def compute_duration(row):
        start_val = row.iloc[start_idx]
        end_val = row.iloc[end_idx]

        if pd.isna(start_val) or pd.isna(end_val) or str(start_val).strip() == "" or str(end_val).strip() == "":
            return ""

        try:
            start_time = pd.to_datetime(start_val)
            end_time = pd.to_datetime(end_val)
            duration_hours = (end_time - start_time).total_seconds() / 3600
            return round(duration_hours, 2)
        except Exception:
            return ""

    data[result_column] = data.apply(compute_duration, axis=1)
    return data

def process_length_of_stay(data, room_col, entry_col, exit_col, step, num_batches, delivery_room_words, max_gap_minutes, result_col):
    """
    Calculates the total length of the most recent continuous stay in a delivery room.
    """
    room_idx = column_name_to_index(data, room_col)
    entry_idx = column_name_to_index(data, entry_col)
    exit_idx = column_name_to_index(data, exit_col)
   
    def process_row(row):
        stays = []
       
        for batch in range(num_batches):
            idx_room = room_idx + (batch * step)
            idx_entry = entry_idx + (batch * step)
            idx_exit = exit_idx + (batch * step)
           
            if pd.notna(row.iloc[idx_room]) and any(word.lower() in str(row.iloc[idx_room]).lower() for word in delivery_room_words):
                if pd.notna(row.iloc[idx_entry]) and pd.notna(row.iloc[idx_exit]):
                    entry_time = pd.to_datetime(row.iloc[idx_entry])
                    exit_time = pd.to_datetime(row.iloc[idx_exit])
                    stays.append((entry_time, exit_time))
       
        stays.sort(reverse=True, key=lambda x: x[0])
        merged_stay_time_minutes = 0.0
       
        if stays:
            current_start, current_end = stays[0]
            merged_stay_time_minutes += (current_end - current_start).total_seconds() / 60
           
            for i in range(1, len(stays)):
                next_start, next_end = stays[i]
                if (current_start - next_end).total_seconds() / 60 <= max_gap_minutes:
                    merged_stay_time_minutes += (next_end - next_start).total_seconds() / 60
                    current_start = next_start
                else:
                    break
       
        return round(merged_stay_time_minutes / 60, 2) if merged_stay_time_minutes > 0 else None
   
    data[result_col] = data.apply(process_row, axis=1)
    return data

def process_length_of_fever(data, date_col, temp_col, step, num_batches, result_col):
    """
    Calculates consecutive fever sequences (temperature above threshold).
    """
    date_idx = column_name_to_index(data, date_col)
    temp_idx = column_name_to_index(data, temp_col)

    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def fever_sequences(row):
        fever_dates = []

        for i in range(num_batches):
            idx_date = date_idx + (i * step)
            idx_temp = temp_idx + (i * step)

            temp_val = safe_float(row.iloc[idx_temp])
            date_val = row.iloc[idx_date]

            if pd.notna(date_val) and temp_val is not None and temp_val >= 38:
                try:
                    fever_dates.append(pd.to_datetime(date_val).date())
                except Exception:
                    pass

        fever_dates = sorted(set(fever_dates))
        if not fever_dates:
            return ""

        sequences = []
        current_streak = 1
        for i in range(1, len(fever_dates)):
            if (fever_dates[i] - fever_dates[i - 1]).days == 1:
                current_streak += 1
            else:
                sequences.append(current_streak)
                current_streak = 1
        sequences.append(current_streak)

        return ", ".join(map(str, sorted(sequences, reverse=True)))

    data[result_col] = data.apply(fever_sequences, axis=1)
    return data

def process_other_cultures(data, collection_date_col, organism_col, specimen_col, step, num_batches, result_samples, result_organisms, result_organism_categories=None, organism_translation_dict=None, specimen_filter_keywords=None):
    """
    Extracts unique sample types and detected organisms from multiple culture test columns.
    """
    collection_idx = column_name_to_index(data, collection_date_col)
    organism_idx = column_name_to_index(data, organism_col)
    specimen_idx = column_name_to_index(data, specimen_col)

    def extract_culture_info(row):
        samples = set()
        organisms = set()
        for i in range(num_batches):
            date_i = collection_idx + (i * step)
            org_i = organism_idx + (i * step)
            spec_i = specimen_idx + (i * step)

            spec_val = str(row.iloc[spec_i]).strip().lower() if pd.notna(row.iloc[spec_i]) else ""

            if specimen_filter_keywords and not any(kw.lower() in spec_val for kw in specimen_filter_keywords):
                continue

            if pd.notna(row.iloc[date_i]) and str(row.iloc[date_i]).strip() != "":
                if spec_val:
                    samples.add(str(row.iloc[spec_i]))
                if pd.notna(row.iloc[org_i]) and str(row.iloc[org_i]).strip() != "":
                    organisms.add(str(row.iloc[org_i]))

        raw_organisms = ', '.join(organisms)
        samples_str = ', '.join(samples)

        if organism_translation_dict and result_organism_categories:
            categories = [organism_translation_dict.get(item.strip(), "") for item in organisms]
            category_str = ', '.join(filter(None, categories))
            return pd.Series([samples_str, raw_organisms, category_str])
        else:
            return pd.Series([samples_str, raw_organisms])

    if organism_translation_dict and result_organism_categories:
        data[[result_samples, result_organisms, result_organism_categories]] = data.apply(extract_culture_info, axis=1)
    else:
        data[[result_samples, result_organisms]] = data.apply(extract_culture_info, axis=1)

    return data

def extract_organism_name_column(data, source_column, target_column, keyword):
    """
    Extracts organism name from a source column based on a keyword and saves it to a target column.
    """
    def extract_if_match(value):
        if isinstance(value, str) and keyword in value:
            return extract_organism_name(value)
        return ""
    
    data[target_column] = data[source_column].apply(extract_if_match)
    return data

def extract_organism_name(text):
    """
    Helper to extract organism name from text (e.g., after ':').
    """
    if pd.isnull(text):
        return ""
    match = re.search(r':\\s*([^,]+)', text)
    if match:
        return match.group(1).strip()
    return text.strip()

def extract_column_value_by_keyword(data, source_column, keyword, target_column, extraction_function=None):
    """
    Extracts a value from a source column based on a keyword and an optional extraction function.
    """
    def extract_value(cell):
        if pd.notna(cell) and keyword in str(cell):
            return extraction_function(cell) if extraction_function else cell
        return ""
    data[target_column] = data[source_column].apply(extract_value)
    return data

def classify_growth_type(text):
    """
    Classifies growth type (no growth, multiple, single) from text.
    """
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    if "no growth" in text or "contaminant" in text:
        return 0
    elif "multiple" in text:
        return 2
    else:
        return 1

def imaging_guided_drainage_detected(data, static_col, repeated_col, step_size, num_steps, keywords, result_col_name, date_col_offset, date_result_col_name):
    """
    Flags rows where imaging-guided drainage was performed and returns the associated timestamp.
    """
    static_idx = column_name_to_index(data, static_col)
    repeated_idx = column_name_to_index(data, repeated_col)
    keywords_lower = [kw.lower() for kw in keywords]

    def check_row(row):
        if pd.notna(row.iloc[static_idx]) and str(row.iloc[static_idx]).strip() != "":
            date_idx = static_idx + date_col_offset
            timestamp = row.iloc[date_idx] if date_idx < len(row) else ""
            return 1, timestamp

        for i in range(num_steps):
            idx = repeated_idx + i * step_size
            if idx >= len(row):
                continue
            val = str(row.iloc[idx]).lower()
            if any(kw in val for kw in keywords_lower):
                date_idx = idx + date_col_offset
                timestamp = row.iloc[date_idx] if date_idx < len(row) else ""
                return 1, timestamp

        return 0, ""

    results = data.apply(check_row, axis=1)
    data[result_col_name] = results.apply(lambda x: x[0])
    data[date_result_col_name] = results.apply(lambda x: x[1])
    return data

def flag_infectious_indication_from_free_text_batch(data, column_name, infectious_phrases, negation_prefixes, result_col, snippet_col, context_window=5, partialMatch=False, batch=1, result_offset=1):
    """
    Flags infectious indications from free text in batches, moving result columns.
    """
    for i in range(1, batch + 1):
        data = flag_infectious_indication_from_free_text(data, f"{column_name}_{i}", infectious_phrases, negation_prefixes, f"{result_col}_{i}", f"{snippet_col}_{i}", context_window, partialMatch)
        data = move_column_relative_to_another(data, f"{column_name}_{i}", result_offset, f"{result_col}_{i}")
        data = move_column_relative_to_another(data, f"{column_name}_{i}", result_offset+1, f"{snippet_col}_{i}")

    return data

def flag_infectious_indication_from_free_text(data, column_name, infectious_phrases, negation_prefixes, result_col, snippet_col, context_window=5, partialMatch=False):
    """
    Flags infectious indications from free text, considering negations and context.
    """
    col_idx = column_name_to_index(data, column_name)
    infectious_phrases_split = [phrase.lower().split() for phrase in infectious_phrases]
    negation_prefixes_lower = [n.lower() for n in negation_prefixes]

    def process_text(text):
        words = re.findall(r'\\b\\w+\\b', str(text).lower())
        for i in range(len(words)):
            for phrase_words in infectious_phrases_split:
                n = len(phrase_words)
                if i + n > len(words):
                    continue

                segment = words[i:i+n]
                if n == 1 and partialMatch:
                    if not any(phrase_words[0] in word for word in segment):
                        continue
                else:
                    if segment != phrase_words:
                        continue

                preceding = words[max(0, i - 3):i]
                if any(neg in preceding for neg in negation_prefixes_lower):
                    continue

                snippet_start = max(0, i - context_window)
                snippet = ' '.join(words[snippet_start:i + n])
                return (1, snippet)
        return (0, '')

    results = data.iloc[:, col_idx].apply(process_text)
    data[result_col] = results.apply(lambda x: x[0])
    data[snippet_col] = results.apply(lambda x: x[1])
    return data

def extract_sentences_containing_words_batch(data, column_name, keywords, negation_prefixes, result_column_name, batch=1, result_offset=1):
    """
    Extracts sentences containing keywords from a column in batches, moving result columns.
    """
    for i in range(1, batch + 1):
        data = extract_sentences_containing_words(data, f"{column_name}_{i}", keywords, negation_prefixes, f"{result_column_name}_{i}")
        data = move_column_relative_to_another(data, f"{column_name}_{i}", result_offset, f"{result_column_name}_{i}")

    return data
    
def extract_sentences_containing_words(data, column_name, keywords, negation_prefixes, result_column_name):
    """
    Extracts dot-separated sentences containing keywords, unless negated.
    """
    col_idx = column_name_to_index(data, column_name)
    keywords_lower = [k.lower() for k in keywords]
    negation_prefixes_lower = [n.lower() for n in negation_prefixes]

    def process_text(text):
        text_lower = str(text).lower()
        sentences = [s.strip() for s in text_lower.split('.') if s.strip()]
        matched_sentences = []

        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                for keyword in keywords_lower:
                    if keyword in word:
                        preceding = words[max(0, i - 2):i]
                        if not any(neg in p for neg in negation_prefixes_lower for p in preceding):
                            matched_sentences.append(sentence)
                            break
                else:
                    continue
                break

        return '. '.join(matched_sentences) if matched_sentences else ''

    data[result_column_name] = data.iloc[:, col_idx].apply(process_text)
    return data

def split_rows_by_non_empty_batches(data, batch_start_col, step_size, num_batches, columns_per_batch, prefix, batch_index_col="CT_Number"):
    """
    Explodes rows by non-empty column batches, creating new rows with standardized column names.
    """
    start_idx = column_name_to_index(data, batch_start_col)
    result_rows = []

    for _, row in data.iterrows():
        base_row = row.to_dict()
        row_split_count = 0

        for batch in range(num_batches):
            batch_indices = [start_idx + batch * step_size + offset for offset in range(columns_per_batch)]

            if not any(pd.notna(row.iloc[idx]) and str(row.iloc[idx]).strip() != '' for idx in batch_indices):
                continue

            row_split_count += 1
            new_row = base_row.copy()

            for idx in batch_indices:
                orig_col_name = data.columns[idx]
                base_name = re.sub(r'[_ ]?\\d+$', '', orig_col_name)
                new_col_name = f"{prefix}{base_name}"
                new_row[new_col_name] = row.iloc[idx]

            new_row[batch_index_col] = row_split_count
            result_rows.append(new_row)

        if row_split_count == 0:
            base_row[batch_index_col] = 0
            result_rows.append(base_row)

    return pd.DataFrame(result_rows)

def check_disinfection_components(data, text_col, scrub_raw_data_col, backup_col, result_col, keyword_dict):
    """
    Checks for presence of 3 disinfection components from multiple sources.
    """
    text_idx = column_name_to_index(data, text_col)
    raw_idx = column_name_to_index(data, scrub_raw_data_col)
    backup_idx = column_name_to_index(data, backup_col)

    lowered_keywords = {
        component: [w.lower() for w in words]
        for component, words in keyword_dict.items()
    }

    def evaluate_row(row):
        found_components = set()

        text_val = str(row.iloc[text_idx]).lower() if pd.notna(row.iloc[text_idx]) else ""
        for component, keywords in lowered_keywords.items():
            if any(kw in text_val for kw in keywords):
                found_components.add(component)

        raw_val = row.iloc[raw_idx]
        if pd.notna(raw_val):
            parts = str(raw_val).split(";")
            for section in parts:
                section = section.strip().lower()
                for component, keywords in lowered_keywords.items():
                    if any(kw in section for kw in keywords):
                        found_components.add(component)

        backup_val = str(row.iloc[backup_idx]).strip() if pd.notna(row.iloc[backup_idx]) else ""

        if len(found_components) == 3:
            return 1
        elif len(found_components) > 0:
            return 0
        elif backup_val == "בוצע":
            return 1
        else:
            return 2

    data[result_col] = data.apply(evaluate_row, axis=1)
    return data

def find_closest_lab_value_batch(data,start_col,step_size,num_batches,date_col_offset,ct_time_reference_col,max_gap_hours_before,result_col,max_gap_hours_after=None,batch=1,result_offset=1):
    """
    Finds the lab value closest to a reference time in batches, moving result columns.
    """
    for i in range(1, batch + 1):
        data = find_closest_lab_value(
            data,
            start_col,
            step_size,
            num_batches,
            date_col_offset,
            f"{ct_time_reference_col}_{i}",
            max_gap_hours_before,
            f"{result_col}_{i}",
            max_gap_hours_after
        )
        data = move_column_relative_to_another(data, f"{ct_time_reference_col}_{i}", result_offset, f"{result_col}_{i}")

    return data

def find_closest_lab_value(
    data,
    start_col,
    step_size,
    num_batches,
    date_col_offset,
    ct_time_reference_col,
    max_gap_hours_before,
    result_col,
    max_gap_hours_after=None
):
    """
    Finds the lab value closest to a reference time (in days from reference).
    """
    start_idx = column_name_to_index(data, start_col)
    ref_idx = column_name_to_index(data, ct_time_reference_col)

    def find_best_match(row):
        try:
            reference_days = float(row.iloc[ref_idx])
        except Exception:
            return ""

        candidates = []

        for i in range(num_batches):
            val_idx = start_idx + i * step_size
            date_idx = val_idx + date_col_offset

            try:
                val = row.iloc[val_idx]
                lab_days = float(row.iloc[date_idx])
                delta_hours = (reference_days - lab_days) * 24
                candidates.append((delta_hours, val))
            except Exception:
                continue

        before = [(abs(d), v) for d, v in candidates if d >= 0 and d <= max_gap_hours_before]
        after = []
        if max_gap_hours_after and max_gap_hours_after > 0:
            after = [(abs(d), v) for d, v in candidates if d < 0 and abs(d) <= max_gap_hours_after]

        if before:
            return sorted(before)[0][1]
        elif after:
            return sorted(after)[0][1]
        else:
            return ""

    data[result_col] = data.apply(find_best_match, axis=1)
    return data

def find_closest_lab_value_by_datetime_batch(data,start_col,step_size,num_batches,date_col_offset,ct_time_col,max_gap_hours_before,result_col,max_gap_hours_after=None,batch=1,result_offset=1):
    """
    Finds the lab value closest to a reference datetime in batches, moving result columns.
    """
    for i in range(1, batch + 1):
        data = find_closest_lab_value_by_datetime(
            data,
            start_col,
            step_size,
            num_batches,
            date_col_offset,
            f"{ct_time_col}_{i}",
            max_gap_hours_before,
            f"{result_col}_{i}",
            max_gap_hours_after
        )
        data = move_column_relative_to_another(data, f"{ct_time_col}_{i}", result_offset, f"{result_col}_{i}")

    return data

def find_closest_lab_value_by_datetime(
    data,
    start_col,
    step_size,
    num_batches,
    date_col_offset,
    ct_time_col,
    max_gap_hours_before,
    result_col,
    max_gap_hours_after=None
):
    """
    Finds the lab value closest to a reference datetime.
    """
    start_idx = column_name_to_index(data, start_col)
    ref_idx = column_name_to_index(data, ct_time_col)

    def find_best_match(row):
        try:
            reference_time = pd.to_datetime(row.iloc[ref_idx])
        except Exception:
            return ""

        candidates = []

        for i in range(num_batches):
            val_idx = start_idx + i * step_size
            date_idx = val_idx + date_col_offset

            try:
                val = row.iloc[val_idx]
                lab_time = pd.to_datetime(row.iloc[date_idx])
                delta_hours = (reference_time - lab_time).total_seconds() / 3600
                candidates.append((delta_hours, val))
            except Exception:
                continue

        before = [(abs(d), v) for d, v in candidates if d >= 0 and d <= max_gap_hours_before]
        after = []
        if max_gap_hours_after and max_gap_hours_after > 0:
            after = [(abs(d), v) for d, v in candidates if d < 0 and abs(d) <= max_gap_hours_after]

        if before:
            return sorted(before)[0][1]
        elif after:
            return sorted(after)[0][1]
        else:
            return ""

    data[result_col] = data.apply(find_best_match, axis=1)
    return data

def detect_multiple_antibiotics(data, source_col, result_col):
    """
    Detects if more than one unique antibiotic name appears in a comma-separated field.
    """
    col_idx = column_name_to_index(data, source_col)

    def extract_names(row):
        raw = row.iloc[col_idx]
        if pd.isna(raw) or str(raw).strip() == "":
            return 0

        cleaned = re.sub(r'(\\d),(\\d{3})', r'\\1\\2', str(raw))

        parts = [p.strip() for p in cleaned.split(",")]
        parts = [p for p in parts if p]

        names = set()
        for part in parts:
            match = re.match(r'^([A-Z]+(?: [A-Z]+)*)', part)
            if match:
                names.add(match.group(1).strip())

        return 1 if len(names) > 1 else 0

    data[result_col] = data.apply(extract_names, axis=1)
    return data

def flag_antibiotic_change_due_to_growth(
    data,
    culture_time_col,
    organism_offset,
    culture_step,
    culture_batches,
    antibiotic_name_col,
    antibiotic_time_offset,
    antibiotic_step,
    antibiotic_batches,
    result_col,
    debug_col,
    max_hours_for_empiric_antibiotic=24,
    min_hours_after_collection_check_antibiotic_change=24,
    max_hours_after_collection_check_antibiotic_change=72
):
    """
    Flags if there was a change in antibiotic treatment following a culture growth.
    """

    def _sanitize_antibiotic_name(name):
        name = str(name).strip()
        match = re.match(r'^([A-Z]+(?: [A-Z]+)*)', name)
        return match.group(1).strip() if match else ""

    culture_time_idx = column_name_to_index(data, culture_time_col)
    antibiotic_name_idx = column_name_to_index(data, antibiotic_name_col)

    def check_row(row):
        abx_events = []
        for i in range(antibiotic_batches):
            name_idx = antibiotic_name_idx + i * antibiotic_step
            time_idx = name_idx + antibiotic_time_offset
            try:
                raw_name = row.iloc[name_idx]
                raw_time = row.iloc[time_idx]
                name = _sanitize_antibiotic_name(raw_name)
                time = float(raw_time) * 24
                if name and pd.notna(time):
                    abx_events.append((time, name))
            except:
                continue

        abx_events.sort()
        debug_matches = []

        for i in range(culture_batches):
            time_idx = culture_time_idx + i * culture_step
            org_idx = time_idx + organism_offset
            try:
                culture_time = float(row.iloc[time_idx]) * 24
                organism = str(row.iloc[org_idx]).strip()
                if organism == "" or pd.isna(organism):
                    continue
            except:
                continue

            empiric_abx = set(
                n for t, n in abx_events
                if abs(t - culture_time) <= max_hours_for_empiric_antibiotic
            )

            if not empiric_abx:
                continue

            after_abx = [
                (t, n) for t, n in abx_events
                if culture_time + min_hours_after_collection_check_antibiotic_change <= t <= culture_time + max_hours_after_collection_check_antibiotic_change
            ]

            changed_abx = [(t, n) for t, n in after_abx if n not in empiric_abx]

            if changed_abx:
                changes_str = "; ".join([f"{n} @ {t - culture_time:.1f}h" for t, n in changed_abx])
                debug_str = (
                    f"growth:{organism} @ {culture_time:.1f}h from ref | "
                    f"empiric:{', '.join(empiric_abx)} | change(s): {changes_str} after culture time"
                )
                debug_matches.append(debug_str)

        if debug_matches:
            return pd.Series([1, " ### ".join(sorted(set(debug_matches)))])
        else:
            return pd.Series([0, ""])

    data[[result_col, debug_col]] = data.apply(check_row, axis=1)
    return data

def concatenate_unique_batches_by_column(data, start_col, step_size, num_batches, element_position, result_col, dedup_columns=None):
    """
    Concatenates unique elements from batches of columns into a new column.
    """
    start_idx = column_name_to_index(data, start_col)

    def process_row(row):
        seen_keys = set()
        extracted_values = []

        for i in range(num_batches):
            batch = []
            for j in range(step_size):
                try:
                    val = row.iloc[start_idx + i * step_size + j]
                except:
                    val = ""
                batch.append(str(val).strip())

            if all(x == "" for x in batch):
                continue

            if dedup_columns:
                try:
                    key = tuple(batch[pos - 1] for pos in dedup_columns)
                except IndexError:
                    continue
            else:
                key = tuple(batch)

            if key not in seen_keys:
                seen_keys.add(key)
                try:
                    extracted_value = batch[element_position - 1]
                    if extracted_value != "":
                        extracted_values.append(extracted_value)
                except IndexError:
                    continue

        return ", ".join(extracted_values)

    data[result_col] = data.apply(process_row, axis=1)
    return data

def move_column_relative_to_another(data, reference_col, offset, results_col):
    """
    Moves a specified column to a new position relative to another column.
    """
    cols = list(data.columns)
    if results_col in cols:
        cols.remove(results_col)
    
    try:
        ref_idx = column_name_to_index(data, reference_col)
    except KeyError:
        print(f"Reference column '{reference_col}' not found. Cannot move '{results_col}'.")
        return data

    new_idx = ref_idx + offset
    
    # Ensure new_idx is within bounds
    if new_idx < 0:
        new_idx = 0
    elif new_idx > len(cols):
        new_idx = len(cols)

    cols.insert(new_idx, results_col)
    return data[cols]
