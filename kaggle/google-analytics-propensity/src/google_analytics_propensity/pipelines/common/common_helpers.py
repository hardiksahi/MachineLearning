import pandas as pd
import inflection


def clean_column_names(column_list):
    column_name_series = pd.Series(column_list)

    ## Step 1: Replace $ with ''
    column_name_series = column_name_series.str.replace(
        r"[\$]", "", regex=True, case=False
    )
    ## Step 2: Replace space with ''
    column_name_series = column_name_series.str.replace(
        r"\s+", "", regex=True, case=False
    )
    ## Step 3: Replace ._ with .
    column_name_series = column_name_series.str.replace(
        r"(\._)", ".", regex=True, case=False
    )
    ## Step 4: Convert Camel Case to snake case using inflection library
    # (https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case/1176023#1176023)

    column_name_series = column_name_series.apply(
        lambda col_name: inflection.underscore(col_name)
    )
    ## Step 5: COnvert columns to lower case
    column_name_series = column_name_series.str.lower()
    ## Step 6: Replace . with _
    column_name_series = column_name_series.str.replace(
        r"(\.)", "_", regex=True, case=False
    )

    return column_name_series.tolist()
