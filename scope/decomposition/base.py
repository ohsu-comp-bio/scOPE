"""Abstract base class for all scOPE decomposition methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from anndata import AnnData


class BaseDecomposition(ABC):
    """Common interface for SVD, NMF, ICA, and PCA decompositions."""

    def __init__(self, n_components: int = 50, layer: str | None = None):
        self.n_components = n_components
        self.layer = layer

    @abstractmethod
    def fit(self, adata: AnnData, y=None) -> BaseDecomposition:
        """Learn decomposition from *adata*."""

    @abstractmethod
    def transform(self, adata: AnnData) -> AnnData:
        """Project *adata* into the learned latent space."""

    def fit_transform(self, adata: AnnData, y=None) -> AnnData:
        """Fit and immediately transform *adata*."""
        return self.fit(adata, y=y).transform(adata)

    @property
    def components_(self) -> np.ndarray:
        """Gene loadings matrix V (genes × components)."""
        raise NotImplementedError

    def _get_X(self, adata: AnnData, layer: str | None = None) -> np.ndarray:
        """Extract the expression matrix as a dense float64 array."""
        import scipy.sparse as sp

        _layer = layer if layer is not None else self.layer
        X = adata.layers[_layer] if _layer else adata.X
        if sp.issparse(X):
            X = X.toarray()
        return X.astype(np.float64)
