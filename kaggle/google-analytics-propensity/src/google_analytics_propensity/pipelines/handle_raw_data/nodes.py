"""
This is a boilerplate pipeline 'handle_raw_data'
generated using Kedro 0.18.3
"""
import pandas as pd
import json
from typing import Any, Dict, Iterator
from datetime import datetime
import logging

from ..common.common_helpers import clean_column_names

logger = logging.getLogger(__name__)


def normalize_dataset(input_dfs: Iterator[pd.DataFrame], params: Dict[str, Any]):
    drop_column_list = params["drop_column_list"]
    json_column_list = params["json_column_list"]

    for i, df in enumerate(input_dfs):
        # ## Step 1: Clean column names
        # df.columns = clean_column_names(df.columns)

        ## Step 1: Drop certain columns
        columns_to_drop_in_df = list(set(drop_column_list).intersection(df.columns))
        df = df.drop(columns_to_drop_in_df, axis=1)

        ## Step 2: Handle json columns
        json_columns_in_df = list(set(json_column_list).intersection(df.columns))
        for json_col in json_columns_in_df:
            expanded_json_column_df = pd.json_normalize(
                df[json_col].apply(lambda x: json.loads(x)).tolist()
            )
            expanded_json_column_df.index = df.index
            expanded_json_column_df.columns = [
                f"{json_col}.{col}" for col in expanded_json_column_df.columns
            ]

            df = df.drop([json_col], axis=1)
            df = df.merge(expanded_json_column_df, left_index=True, right_index=True)

        # ## Step 3: Get number of hits from hit list
        # df.loc[:, "hits_column_length"] = df["hits"].apply(lambda ll: len(ll))
        # df = df.drop(["hits"], axis=1)

        ## Step 3: Set datatype of fullVisitorId as str
        column_as_string_list = params["column_as_string_list"]
        for col in column_as_string_list:
            df.loc[:, col] = df[col].astype(str)

        ## Step 4: Clean column names:
        df.columns = clean_column_names(df.columns)

        ## Step 5: Get unique session identifer
        df.loc[:, "unique_session_identifier"] = (
            df["full_visitor_id"].astype(str) + "_" + df["visit_id"].astype(str)
        )

        ## Step 6: Get utc based date time
        df.loc[:, "utc_visit_start_time"] = pd.to_datetime(
            df["visit_start_time"], unit="s"
        )

        ## Step 7: Convert date column to pandas datatime
        df.loc[:, "date_object"] = df["date"].apply(
            lambda d: datetime.strptime(str(d), "%Y%m%d")
        )

        logger.info(f"Saving {i}th subset to disk")
        yield {f"subset_{i}": df}
