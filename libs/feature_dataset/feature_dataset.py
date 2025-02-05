from typing import Any, List, Optional
import pandas as pd
import duckdb
import os

DEFAULT_FEATURE_STORE_FILE = "data/feature_store.duckdb"


class FeatureDataset:
    def __init__(self, database_file: Optional[str] = None):
        self._database_file = database_file or DEFAULT_FEATURE_STORE_FILE

    def get_feature_dataset(
        self, name: str, features: Optional[List[str]] = None
    ) -> duckdb.DuckDBPyRelation:
        con = duckdb.connect(database=self._database_file)
        if features:
            features_str = ", ".join(features)
        else:
            features_str = "*"
        query = f"SELECT {features_str} FROM fd_{name}_1"
        data = con.query(query).to_df()
        con.close()
        return data

    def create_feature_dataset(
        self, name: str, dataset: pd.DataFrame, version: int = 1
    ):
        con = duckdb.connect(database=self._database_file)
        con.execute(
            f"CREATE OR REPLACE TABLE fd_{name}_{version} AS SELECT * FROM dataset"
        )
        con.close()
