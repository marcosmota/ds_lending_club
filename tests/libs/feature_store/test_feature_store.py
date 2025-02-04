import os
import pytest
import pandas as pd
from feature_dataset.feature_dataset import FeatureStore

TEST_DB_FILE = "test_feature_store.duckdb"

@pytest.fixture(scope="module")
def feature_store():
    fs = FeatureStore(TEST_DB_FILE)
    yield fs
    os.remove(TEST_DB_FILE)

def test_create_feature_dataset(feature_store):
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    df = pd.DataFrame(data)

    feature_store.create_feature_dataset("test", df)

    result_df = feature_store.get_feature_dataset("test").to_df()

    pd.testing.assert_frame_equal(df, result_df)

def test_get_feature_dataset_with_specific_features(feature_store):
    data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
    df = pd.DataFrame(data)

    feature_store.create_feature_dataset("test", df)

    result_df = feature_store.get_feature_dataset("test", features=["feature1"]).to_df()
    pd.testing.assert_frame_equal(df.filter(["feature1"]), result_df)
