

import pandas as pd
from collections import defaultdict
import re
from backend.functions.utilities import column_name_to_index

def generate_heatmap_with_counts(data, start_column, columns_per_set, num_tuples, allow_multiple_duplicates=False, prefix_delimiter=" - ", output_file="heatmap.csv"):
    """
    Generate a heatmap matrix of unique Col1 (columns) and Col2 (rows), counting values from Col3.
    """
    heatmap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    start_index = column_name_to_index(data, start_column) if isinstance(start_column, str) else start_column

    for idx, row in data.iterrows():
        seen_combinations = {}

        for i in range(num_tuples):
            col1_index = start_index + i * columns_per_set
            col2_index = col1_index + 1
            col3_index = col1_index + 2
            col5_index = col1_index + 4 if columns_per_set >= 5 else None

            if col3_index >= len(data.columns):
                break

            col1_value = row.iloc[col1_index]
            col2_value = row.iloc[col2_index]
            col3_value = row.iloc[col3_index]
            col5_value = row.iloc[col5_index] if col5_index and col5_index < len(data.columns) else None

            col1_value = col1_value if pd.notna(col1_value) and col1_value != "" else "Empty"
            col2_value = col2_value if pd.notna(col2_value) and col2_value != "" else "Empty"
            col3_value = col3_value if pd.notna(col3_value) and col3_value != "" else "Empty"
            col5_value = col5_value if pd.notna(col5_value) and col5_value != "" else ""

            if col5_value:
                if col2_value != "Empty":
                    col2_value = f"{col5_value}{prefix_delimiter}{col2_value}"
                else:
                    col2_value = col5_value

            combination_key = (col1_value, col2_value)
            if combination_key in seen_combinations:
                if not allow_multiple_duplicates:
                    continue
            else:
                seen_combinations[combination_key] = col3_value

            heatmap[col2_value][col1_value][col3_value] += 1

    unique_cols = sorted({col for row_dict in heatmap.values() for col in row_dict.keys()})
    unique_rows = sorted(heatmap.keys())
    heatmap_dict = {col1: [] for col1 in unique_cols}

    for col2_value in unique_rows:
        for col1_value in unique_cols:
            count_dict = heatmap[col2_value][col1_value]
            count_str = ", ".join(f"{k}: {v}" for k, v in count_dict.items()) if count_dict else ""
            heatmap_dict[col1_value].append(count_str)

    heatmap_df = pd.DataFrame(heatmap_dict, index=unique_rows)
    heatmap_df.to_csv(output_file, index_label="Organism \\ Antibiotic")
    return heatmap_df

def generate_patient_specific_dataset(data, start_column, columns_per_set, num_tuples, patient_id_column, additional_fields=[], output_file="patient_dataset.csv"):
    """
    Generate a dataset where each row represents a unique Virus (Col2 value) per patient,
    mapping Antibiotic (Col1) to their Susceptibility (Col3) values.
    """
    start_index = column_name_to_index(data, start_column) if isinstance(start_column, str) else start_column
    patient_data = []

    for idx, row in data.iterrows():
        patient_id = row[patient_id_column]
        patient_row = {field: row[field] for field in additional_fields}
        patient_row["PatientId"] = patient_id

        virus_map = {}
        patient_map = defaultdict(lambda: defaultdict(str))

        for i in range(num_tuples):
            virus_index = start_index + i * columns_per_set + 1
            antibiotic_index = start_index + i * columns_per_set
            susceptibility_index = start_index + i * columns_per_set + 2
            alternative_virus_index = start_index + i * columns_per_set + 4

            if alternative_virus_index >= len(data.columns):
                break

            virus_value = row.iloc[virus_index] if pd.notna(row.iloc[virus_index]) and row.iloc[virus_index] != "" else None
            antibiotic_value = row.iloc[antibiotic_index] if pd.notna(row.iloc[antibiotic_index]) and row.iloc[antibiotic_index] != "" else None
            susceptibility_value = row.iloc[susceptibility_index] if pd.notna(row.iloc[susceptibility_index]) and row.iloc[susceptibility_index] != "" else None
            alternative_virus_value = row.iloc[alternative_virus_index] if pd.notna(row.iloc[alternative_virus_index]) and row.iloc[alternative_virus_index] != "" else None

            if not virus_value:
                continue

            if virus_value not in virus_map and alternative_virus_value:
                virus_map[virus_value] = alternative_virus_value

            if not antibiotic_value:
                if virus_value not in patient_map:
                    patient_map[virus_value] = {}
                continue

            if susceptibility_value:
                if antibiotic_value not in patient_map[virus_value]:
                    patient_map[virus_value][antibiotic_value] = susceptibility_value
                else:
                    patient_map[virus_value][antibiotic_value] += f", {susceptibility_value}"

        for virus_value, antibiotic_map in patient_map.items():
            new_row = {
                "PatientId": patient_id,
                "Virus": virus_value,
                "AlternativeVirusName": virus_map.get(virus_value, ""),
            }

            for key, value in patient_row.items():
                new_row[key] = value

            for antibiotic_key, susceptibility_values in antibiotic_map.items():
                new_row[antibiotic_key] = susceptibility_values

            patient_data.append(new_row)

    result_df = pd.DataFrame(patient_data)
    result_df = result_df.fillna("")
    result_df.to_csv(output_file, index=False)
    return result_df

def summarize_keys_and_values_in_raw_map(data: pd.DataFrame, input_column: str, output_file: str) -> pd.DataFrame:
    """
    Summarize unique keys and their corresponding unique values in raw map-like data.
    """
    key_value_map = defaultdict(set)

    def process_row(row):
        tokens = row.split(";")
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

            key_value_map[key].add(value)

    data[input_column].apply(process_row)

    max_values = max(len(values) for values in key_value_map.values()) if key_value_map else 0
    summary_data = {}

    for key, values in key_value_map.items():
        summary_data[key] = list(values) + [""] * (max_values - len(values))

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    return summary_df

