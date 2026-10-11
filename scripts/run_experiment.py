from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cf_model import CFConfig, MeshCollaborativeFiltering
from src.data_utils import extract_terms_from_query, filter_vocabulary, load_csv, query_vector_from_terms
from src.evaluate import evaluate_model

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def run_once(df, seed: int, min_df: int = 1, max_terms: int = 100, epochs: int = 40, batch_size: int = 64):
    df2, vocab = filter_vocabulary(df, min_doc_freq=min_df, max_terms=max_terms)
    terms = df2["mesh_terms_list"].tolist()
    U, vocab2 = MeshCollaborativeFiltering.tfidf_matrix(terms)

    rng = np.random.default_rng(seed)
    idx = np.arange(U.shape[1])
    rng.shuffle(idx)
    n = len(idx)
    if n < 3:
        raise ValueError("At least 3 usable documents are required for train/dev/test evaluation.")
    n_train = max(1, int(round(0.8 * n)))
    n_dev = max(1, int(round(0.1 * n)))
    if n_train + n_dev >= n:
        n_dev = 1
        n_train = max(1, n - 2)
    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]
    test_idx = idx[n_train + n_dev:]

    R = U[:, train_idx]
    Xdev = U[:, dev_idx]
    Xtest = U[:, test_idx]
    cfg = CFConfig(
        tau=-5.0,
        k=1.0,
        learning_rate=0.01,
        reg_lambda=0.001,
        epochs=epochs,
        batch_size=min(batch_size, R.shape[1]),
        seed=seed,
    )
    model = MeshCollaborativeFiltering(cfg).fit(R)
    dev = evaluate_model(model, R, Xdev, seed=seed + 1)
    test = evaluate_model(model, R, Xtest, seed=seed + 2)
    result = {
        "seed": seed,
        "documents": int(len(df2)),
        "terms": int(len(vocab2)),
        "train_docs": int(len(train_idx)),
        "dev_docs": int(len(dev_idx)),
        "test_docs": int(len(test_idx)),
        "dev_aptk": dev.model_aptk,
        "dev_popularity_aptk": dev.popularity_aptk,
        "dev_random_aptk": dev.random_aptk,
        "dev_mser": dev.mser,
        "test_aptk": test.model_aptk,
        "test_popularity_aptk": test.popularity_aptk,
        "test_random_aptk": test.random_aptk,
        "test_mser": test.mser,
        "final_loss": float(model.loss_history_[-1]),
    }
    return result, model, U, vocab2, df2


def write_query_examples(model, U, vocab, output_path: Path):
    doc_freq = (U > 0).sum(axis=1).astype(np.float32)
    queries = [
        "Clinical trial involving Arthritis, Rheumatoid and autoimmune diseases with methotrexate",
        "Patients with Lung Neoplasms receiving cancer treatment",
        "Study of postoperative complications after hernia surgery",
    ]
    examples = []
    for query in queries:
        terms = extract_terms_from_query(query, vocab)
        vector = query_vector_from_terms(terms, vocab, n_docs=U.shape[1], doc_freq=doc_freq)
        recs = model.recommend(vector, vocab, top_k=3)
        examples.append((query, terms, recs))

    output_path.write_text(
        "\n\n".join(
            f"QUERY: {query}\nMATCHED: {terms}\nRECOMMENDED: {recs}"
            for query, terms, recs in examples
        ),
        encoding="utf-8",
    )
    return examples


def main():
    parser = argparse.ArgumentParser(description="Run the clinical-trial MeSH collaborative-filtering experiment.")
    parser.add_argument("--input", default=str(ROOT / "data" / "sample_trials.csv"))
    parser.add_argument("--min-doc-freq", type=int, default=1)
    parser.add_argument("--max-terms", type=int, default=100)
    parser.add_argument("--max-documents", type=int, default=0,
                        help="Optional deterministic cap after reading the input CSV; 0 uses all rows")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    df = load_csv(args.input)
    if args.max_documents > 0 and len(df) > args.max_documents:
        df = df.sample(n=args.max_documents, random_state=42).reset_index(drop=True)
    if len(df) < 3:
        parser.error("Input must have at least 3 records containing MeSH terms.")

    rows = []
    last = None
    for seed in args.seeds:
        row, model, U, vocab, df2 = run_once(
            df, seed, args.min_doc_freq, args.max_terms, args.epochs, args.batch_size
        )
        rows.append(row)
        last = (model, U, vocab, df2)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "metrics_by_seed.csv", index=False)
    metrics.agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"}).to_csv(
        RESULTS / "metrics_summary.csv", index=False
    )

    model, U, vocab, df2 = last
    examples = write_query_examples(model, U, vocab, RESULTS / "query_examples.txt")
    summary_lines = [
        f"Input: {args.input}",
        f"Rows used before vocabulary filtering: {len(df)}",
        f"Filtered documents: {len(df2)}",
        f"Vocabulary terms: {len(vocab)}",
        f"Seeds: {args.seeds}",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch_size}",
        f"Mean test APTK@3: {metrics.test_aptk.mean():.4f}",
        f"Mean popularity APTK@3: {metrics.test_popularity_aptk.mean():.4f}",
        f"Mean random APTK@3: {metrics.test_random_aptk.mean():.4f}",
        f"Mean test MSER summary: {metrics.test_mser.mean():.6f}",
        "",
        "Query examples:",
    ]
    for query, terms, recs in examples:
        summary_lines.append(f"- {query}")
        summary_lines.append(f"  matched: {', '.join(terms) if terms else 'none'}")
        summary_lines.append("  recommendations: " + "; ".join(f"{term} ({score:.4f})" for term, score in recs))
    (RESULTS / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(metrics.to_string(index=False))
    print("\nMEAN TEST APTK@3:", metrics.test_aptk.mean())
    print("MEAN POPULARITY BASELINE:", metrics.test_popularity_aptk.mean())
    print("MEAN RANDOM BASELINE:", metrics.test_random_aptk.mean())
    print("MEAN TEST MSER SUMMARY:", metrics.test_mser.mean())


if __name__ == "__main__":
    main()
