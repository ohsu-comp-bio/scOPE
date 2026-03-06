"""Abstract base class for all scOPE decomposition methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


class BaseDecomposition(ABC):
    """Common interface for SVD, NMF, ICA, and PCA decompositions."""

    def __init__(self, n_components: int = 50, layer: str | None = None):
        self.n_components = n_components
        self.layer = layer
        # Subclasses set this in fit():
        self.components_: np.ndarray  # (n_components, n_genes)

    @abstractmethod
    def fit(self, adata: AnnData, y=None) -> BaseDecomposition:
        """Learn decomposition from *adata*."""

    @abstractmethod
    def transform(self, adata: AnnData) -> AnnData:
        """Project *adata* into the learned latent space."""

    def fit_transform(self, adata: AnnData, y=None) -> AnnData:
        """Fit and immediately transform *adata*."""
        return self.fit(adata, y=y).transform(adata)

    def _get_X(self, adata: AnnData, layer: str | None = None):
        """Extract the expression matrix.

        Returns a sparse matrix if the input is sparse, otherwise a dense
        float64 array. Callers that need dense should call ``.toarray()``
        themselves; the SVD/PCA/ICA/NMF transform methods handle this via
        ``_get_X_dense``.
        """
        _layer = layer if layer is not None else self.layer
        X = adata.layers[_layer] if _layer else adata.X
        if sp.issparse(X):
            return X.astype(np.float64)
        return np.asarray(X, dtype=np.float64)

    def _get_X_dense(self, adata: AnnData, layer: str | None = None) -> np.ndarray:
        """Extract the expression matrix as a dense float64 array.

        Use this in ``fit()`` where a dense array is required (e.g. sklearn
        decompositions). For ``transform()`` on large sc datasets, prefer
        ``_get_X`` to avoid unnecessary densification.
        """
        X = self._get_X(adata, layer=layer)
        if sp.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)
