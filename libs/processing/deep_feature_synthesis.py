from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
import featuretools as ft
from featuretools.primitives.base.primitive_base import PrimitiveBase


class DeepFeatureSynthesisPipeline(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        name: str,
        aggs_primitives: List[str] = [],
        windows_primitive: List[PrimitiveBase] = [],
    ):
        self._windows_primitive = windows_primitive
        self._aggs_primitives = aggs_primitives
        self._name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        es = ft.EntitySet(id=self._name)
        es = es.add_dataframe(
            dataframe_name=self._name,
            dataframe=X,
        )

        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name=self._name,
            trans_primitives=self._windows_primitive,
            agg_primitives=self._aggs_primitives,
            verbose=True,
        )

        feature_matrix.columns = feature_matrix.columns.str.lower()

        self.feature_matrix = feature_matrix
        self.feature_defs = feature_defs
        return feature_matrix
