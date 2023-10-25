import pandas as pd
import inflection
import pyarrow as pa
from typing import Any, List, Union


def clean_column_names(column_list: List[str]):
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


def get_appropriate_pyarrow_class(datatype_string: str) -> Union[pa.DataType, Any]:
    pyarrow_datatype_map = {
        "string": pa.string(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "uint8": pa.uint8(),
        "uint32": pa.uint32(),
        "uint64": pa.uint64(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "bool": pa.bool_(),
        "list_of_string": pa.list_(pa.string()),
        "timestamp_utc": pa.timestamp("us", tz="UTC"),
        "time64": pa.time64("us"),
    }
    return pyarrow_datatype_map[datatype_string]
