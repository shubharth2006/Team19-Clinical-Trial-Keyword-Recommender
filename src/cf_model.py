"""Collaborative filtering model for MeSH keyword recommendation.

Implements the core modelling idea described by Zhou (2020):
- TF-IDF utility matrix U over MeSH terms x trial documents
- neighbor matrix W based on cosine similarity
- ReLU + L2-regularized collaborative filtering model
- mini-batch gradient descent
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CFConfig:
    tau: float = -5.0
    k: float = 1.0
    learning_rate: float = 0.01
    reg_lambda: float = 0.001
    epochs: int = 60
    batch_size: int = 16
    seed: int = 42


class MeshCollaborativeFiltering:
    def __init__(self, config: CFConfig | None = None):
        self.config = config or CFConfig()
        self.W_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    @staticmethod
    def tfidf_matrix(document_terms: list[Iterable[str]]) -> tuple[np.ndarray, list[str]]:
        """Build Zhou-style log-scaled TF-IDF matrix (terms x documents).

        This is dense and intended for controlled vocabularies and manageable
        document counts. For the full dataset, use a bounded sample or replace
        this builder with a sparse/distributed implementation.
        """
        vocab = sorted({str(t).strip() for terms in document_terms for t in terms if str(t).strip()})
        term_to_idx = {t: i for i, t in enumerate(vocab)}
        m, n = len(vocab), len(document_terms)
        counts = np.zeros((m, n), dtype=np.float32)
        for j, terms in enumerate(document_terms):
            for term in set(str(t).strip() for t in terms if str(t).strip()):
                if term in term_to_idx:
                    counts[term_to_idx[term], j] += 1.0
        df = (counts > 0).sum(axis=1).astype(np.float32)
        n_docs = max(n, 1)
        idf = np.log10(np.maximum(n_docs, 1) / np.maximum(df, 1.0))
        return np.log10(1.0 + counts) * idf[:, None], vocab

    @staticmethod
    def build_W(R: np.ndarray, tau: float, k: float) -> np.ndarray:
        sim = cosine_similarity(R)
        dist = np.clip(1.0 - sim, 0.0, 1.0)
        W = np.exp(tau * np.power(dist, k)).astype(np.float32)
        np.fill_diagonal(W, 0.0)
        return W

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def fit(self, R: np.ndarray, verbose: bool = False) -> "MeshCollaborativeFiltering":
        if R.ndim != 2:
            raise ValueError("R must be a 2D matrix: terms x documents")
        m, n = R.shape
        if m == 0 or n == 0:
            raise ValueError("R must contain at least one term and one document")
        rng = np.random.default_rng(self.config.seed)
        self.W_ = self.build_W(R, self.config.tau, self.config.k)
        self.theta_ = np.zeros((m, m), dtype=np.float32)
        self.bias_ = np.zeros((m, 1), dtype=np.float32)
        self.loss_history_ = []

        indices = np.arange(n)
        for epoch in range(self.config.epochs):
            rng.shuffle(indices)
            for start in range(0, n, max(1, self.config.batch_size)):
                batch_idx = indices[start : start + max(1, self.config.batch_size)]
                B = R[:, batch_idx]
                raw = (self.W_ * self.theta_) @ B + self.bias_
                pred = self.relu(raw)
                err = pred - B
                grad_mask = (raw >= 0).astype(np.float32)
                gated_err = err * grad_mask
                denom = max(len(batch_idx), 1)
                grad_theta = (self.W_ * (gated_err @ B.T)) / denom
                grad_theta += self.config.reg_lambda * self.theta_
                grad_bias = gated_err.sum(axis=1, keepdims=True) / denom
                self.theta_ -= self.config.learning_rate * grad_theta
                self.bias_ -= self.config.learning_rate * grad_bias

            pred_all = self.predict_matrix(R)
            loss = 0.5 * float(np.mean((pred_all - R) ** 2))
            self.loss_history_.append(loss)
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"epoch={epoch + 1:03d} loss={loss:.6f}")
        return self

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None or self.theta_ is None or self.bias_ is None:
            raise RuntimeError("Model is not fitted")
        if X.shape[0] != self.theta_.shape[0]:
            raise ValueError("Input row count must match the fitted vocabulary")
        return self.relu((self.W_ * self.theta_) @ X + self.bias_)

    def recommend_scores(self, vector: np.ndarray, exclude_observed: bool = True) -> np.ndarray:
        scores = self.predict_matrix(vector.reshape(-1, 1)).ravel()
        if exclude_observed:
            scores[vector > 0] = -np.inf
        return scores

    def recommend(self, vector: np.ndarray, vocabulary: list[str], top_k: int = 3) -> list[tuple[str, float]]:
        scores = self.recommend_scores(vector, exclude_observed=True)
        order = np.argsort(scores)[::-1]
        return [(vocabulary[i], float(scores[i])) for i in order[:top_k] if np.isfinite(scores[i])]


def mask_matrix(X: np.ndarray, fraction: float = 0.5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Randomly mask a fraction of positive entries in each document."""
    rng = np.random.default_rng(seed)
    X_masked = X.copy()
    mask = np.zeros_like(X, dtype=bool)
    for j in range(X.shape[1]):
        pos = np.flatnonzero(X[:, j] > 0)
        if len(pos) == 0:
            continue
        count = max(1, int(round(len(pos) * fraction)))
        chosen = rng.choice(pos, size=min(count, len(pos)), replace=False)
        X_masked[chosen, j] = 0.0
        mask[chosen, j] = True
    return X_masked, mask
