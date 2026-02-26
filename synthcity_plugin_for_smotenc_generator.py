# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema


class SmoteNCPlugin(Plugin):
    """
    Implements the SMOTE algorithm for synthetic data generation.
    
    Args:
        k: Number of nearest neighbors to use
    """
    
    def __init__(
        self,
        k: int = 5,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.k = k
    
    @staticmethod
    def name() -> str:
        return "smotenc"

    @staticmethod
    def type() -> str:
        return "generic"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "SmoteNCPlugin":
        self.synthX = synth_smotenc(
            dat=X.dataframe(), 
            k=self.k
        )
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> pd.DataFrame:
            synthetic_data = self.synthX
            return synthetic_data

        return self._safe_generate(_sample, count, syn_schema)