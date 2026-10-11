from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cf_model import MeshCollaborativeFiltering, mask_matrix


@dataclass
class EvalResult:
    model_aptk: float
    popularity_aptk: float
    random_aptk: float
    mser: float


def _topk_precision(y_true: np.ndarray, candidates: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(candidates) == 0:
        return 0.0
    ranked = candidates[np.argsort(scores[candidates])[::-1]][:k]
    return float(np.sum(y_true[ranked] > 0) / max(k, 1))


def evaluate_model(
    model: MeshCollaborativeFiltering,
    train_R: np.ndarray,
    X: np.ndarray,
    mask_fraction: float = 0.5,
    k: int = 3,
    seed: int = 42,
) -> EvalResult:
    """Evaluate recovery of randomly hidden positive terms in held-out documents.

    MSER here is the implementation's masked-score squared-error summary. It is
    logged as a project metric and should not be described as an exact replication
    of a published benchmark unless its definition is verified against the paper.
    """
    X_masked, _ = mask_matrix(X, mask_fraction, seed)
    pred = model.predict_matrix(X_masked)
    model_hits, pop_hits, random_hits = [], [], []
    popularity = (train_R > 0).sum(axis=1).astype(float)
    rng = np.random.default_rng(seed)

    for j in range(X.shape[1]):
        candidates = np.flatnonzero(X_masked[:, j] == 0)
        if len(candidates) == 0:
            continue
        model_hits.append(_topk_precision(X[:, j], candidates, pred[:, j], k))
        pop_hits.append(_topk_precision(X[:, j], candidates, popularity, k))
        random_candidates = rng.permutation(candidates)[:k]
        random_hits.append(float(np.sum(X[random_candidates, j] > 0) / max(k, 1)))

    relevant = (X > 0) | (pred > 0)
    mser = float(np.sum((pred - X) ** 2 * relevant) / max(int(np.sum(relevant)), 1))
    return EvalResult(
        model_aptk=float(np.mean(model_hits) if model_hits else 0.0),
        popularity_aptk=float(np.mean(pop_hits) if pop_hits else 0.0),
        random_aptk=float(np.mean(random_hits) if random_hits else 0.0),
        mser=mser,
    )
