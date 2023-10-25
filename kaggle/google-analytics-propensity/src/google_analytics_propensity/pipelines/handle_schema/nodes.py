"""
This is a boilerplate pipeline 'handle_schema'
generated using Kedro 0.18.12
"""
import pandas as pd
import json
import logging
import pyarrow as pa
from typing import Any, Dict, Iterator, Callable
from ..common.common_helpers import get_appropriate_pyarrow_class

logger = logging.getLogger(__name__)


def perform_schema_maintenance(
    df: pd.DataFrame, relevant_info_dict: Dict[str, Any], identifier: str
) -> pa.Table:
    logger.info(f"[{identifier}]: Start impose_schema")
    col_name_data_type_dict = relevant_info_dict.get("col_name_data_type_dict")
    universal_column_list = relevant_info_dict.get("universal_column_list")
    missing_columns = list(set(universal_column_list).difference(df.columns))
    logger.info(
        f"[{identifier}]: Adding missing columns: {missing_columns} to parquet"
        if len(missing_columns) > 0
        else f"[{identifier}]: No columns missing in parquet file."
    )

    ## Setting missing_columns as None
    df.loc[:, missing_columns] = None

    ## Ensure that order of columns is same as universal_column_list
    df = df[universal_column_list]

    ## Convert pandas df to pyarrow table
    parquet_table = pa.Table.from_pandas(df, preserve_index=False)
    col_list = [col_schema.name for col_schema in parquet_table.schema]
    fields = [
        pa.field(col_name, col_name_data_type_dict.get(col_name))
        for col_name in col_list
    ]

    ## Enforce schema
    enforced_schema = pa.schema(fields)
    parquet_table = parquet_table.cast(enforced_schema)
    logger.info(f"{identifier}: Successfully imposed schema")

    return parquet_table


def impose_schema(input_dfs: Iterator[pd.DataFrame], params: Dict[str, Any]):
    logger.info("[START] impose_schema")
    # enforced_column_name_data_type_string_dict = json.load(
    #     params.get("enforced_column_name_type_dict_path", None)
    # )
    json_mapping_path = params.get("enforced_column_name_type_dict_path", None)
    column_superset_list_path = params.get("column_superset_list")

    with open(column_superset_list_path, "r") as f:
        universal_column_list = f.read().split("\n")

    with open(json_mapping_path, "r") as j:
        enforced_column_name_data_type_string_dict = json.loads(j.read())

    col_name_data_type_dict = {
        col_name: get_appropriate_pyarrow_class(data_type_string)
        for col_name, data_type_string in enforced_column_name_data_type_string_dict.items()
    }

    relevant_information_dict = {
        "col_name_data_type_dict": col_name_data_type_dict,
        "universal_column_list": universal_column_list,
    }

    result_dict = {}
    for identifier, input_func in input_dfs.items():
        df = input_func() if callable(input_func) else input_func
        parquet_table = perform_schema_maintenance(
            df, relevant_information_dict, identifier
        )
        result_dict[identifier] = parquet_table
        # yield {identifier: parquet_table}

    logger.info("[END] impose_schema")
    return result_dict


def clean_data(
    schema_imposed_data: Dict[str, Callable[[], Any]], params: Dict[str, Any]
):
    logger.info("[START] clean_data")
    drop_column_list = params.get("drop_column_list")

    result_dict = {}
    for identifier, input_func in schema_imposed_data.items():
        parquet_table = input_func() if callable(input_func) else input_func
        logger.info(
            f"[{identifier}] Shape of parquet_table before dropping columns: ({parquet_table.num_rows}, {parquet_table.num_columns})"
        )
        parquet_table = parquet_table.drop(drop_column_list)
        logger.info(
            f"[{identifier}] Shape of parquet_table after dropping columns: ({parquet_table.num_rows}, {parquet_table.num_columns})"
        )
        result_dict[identifier] = parquet_table
    logger.info("[END] clean_data")
    return result_dict
