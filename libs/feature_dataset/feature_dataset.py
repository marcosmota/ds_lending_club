from typing import Any, List, Optional
import pandas as pd
import duckdb
import os

DEFAULT_FEATURE_STORE_FILE = "data/feature_store.duckdb"


class FeatureDataset:
    def __init__(self, database_file: Optional[str] = None):
        self._con = duckdb.connect(
            database=(database_file or DEFAULT_FEATURE_STORE_FILE)
        )

    def get_feature_dataset(
        self, name: str, features: Optional[List[str]] = None
    ) -> duckdb.DuckDBPyRelation:
        if features:
            features_str = ", ".join(features)
        else:
            features_str = "*"
        query = f"SELECT {features_str} FROM fd_{name}_1"
        return self._con.query(query)

    def create_feature_dataset(
        self, name: str, dataset: pd.DataFrame, version: int = 1
    ):
        self._con.execute(
            f"CREATE OR REPLACE TABLE fd_{name}_{version} AS SELECT * FROM dataset"
        )
