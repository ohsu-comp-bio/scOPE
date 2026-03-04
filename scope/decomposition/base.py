"""Abstract base class for all scOPE decomposition methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from anndata import AnnData


class BaseDecomposition(ABC):
    """Common interface for SVD, NMF, ICA, and PCA decompositions.

    All decompositions follow the scikit-learn transformer convention:
    ``fit`` → ``transform`` → ``fit_transform``.
    """

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
        """Extract the expression matrix from *adata* as a dense float64 array."""
        import scipy.sparse as sp

        X = adata.layers[layer] if layer else adata.X
        if sp.issparse(X):
            X = X.toarray()
        return X.astype(np.float64)
