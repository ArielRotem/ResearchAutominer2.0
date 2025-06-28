

def detect_combination_antibiotics(data, source_col, result_col_2plus, result_col_3plus, combos, synonyms, ignored=[]):
    """
    Detects combinations of antibiotics in a comma-separated source column.
    Creates two new columns: one for 2+ unique antibiotics, and one for 3+ unique antibiotics.
    Handles synonyms and ignored antibiotics.
    """
    def _normalize_name(name):
        name = str(name).strip().upper()
        for syn_list in synonyms:
            if name in syn_list:
                return syn_list[0] # Use the first synonym as the canonical name
        return name

    def process_antibiotics(cell):
        if pd.isna(cell) or str(cell).strip() == "":
            return 0, 0, []

        raw_abx = [a.strip() for a in str(cell).split(',') if a.strip()]
        
        normalized_abx = set()
        for abx in raw_abx:
            norm_abx = _normalize_name(abx)
            if norm_abx and norm_abx not in ignored:
                normalized_abx.add(norm_abx)

        num_unique = len(normalized_abx)
        
        has_2plus = 1 if num_unique >= 2 else 0
        has_3plus = 1 if num_unique >= 3 else 0

        return has_2plus, has_3plus, list(normalized_abx)

    results = data[source_col].apply(process_antibiotics)
    data[result_col_2plus] = results.apply(lambda x: x[0])
    data[result_col_3plus] = results.apply(lambda x: x[1])
    # You might want to return the list of normalized_abx for debugging or further use
    # data['normalized_abx_list'] = results.apply(lambda x: x[2])
    return data
