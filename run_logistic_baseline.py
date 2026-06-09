"""
run_logistic_baseline.py
========================
Logistic-regression baselines for DNA sequence classification.

Sequences are featurised using one or more of:
  - k-mer frequency vectors  (default: 3-mer, 4-mer, 6-mer)
  - GC content               (single scalar)

Results are written to a CSV and printed to stdout, making it easy to
compare directly against DNABERT-2 fine-tuning results.

Usage
-----
    python run_logistic_baseline.py \\
        --train_csv  data/train.csv \\
        --test_csv   data/test.csv  \\
        --output_csv results/baseline_results.csv \\
        --kmers      3,4,6 \\
        --max_iter   1000

    # Compare against DNABERT-2 eval_results.json
    python run_logistic_baseline.py \\
        --train_csv       data/train.csv \\
        --test_csv        data/test.csv  \\
        --output_csv      results/baseline_results.csv \\
        --dnabert_results output/dnabert2/results/DNABERT2_run/eval_results.json
"""

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def kmer_counts(sequence: str, k: int) -> dict[str, int]:
    """Return a dict of k-mer → raw count for `sequence`."""
    counts: dict[str, int] = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def build_kmer_vocabulary(sequences: list[str], k: int) -> list[str]:
    """Return a sorted list of all k-mers observed in `sequences`."""
    vocab: set[str] = set()
    for seq in sequences:
        vocab.update(kmer_counts(seq, k).keys())
    return sorted(vocab)


def featurise_kmer(
    sequences: list[str], k: int, vocab: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """
    Convert sequences to a normalised k-mer frequency matrix.

    Parameters
    ----------
    sequences : list of str
    k : int
    vocab : list of str or None
        If None, vocabulary is built from `sequences`.

    Returns
    -------
    X : np.ndarray, shape (n_sequences, len(vocab))
    vocab : list of str
    """
    if vocab is None:
        vocab = build_kmer_vocabulary(sequences, k)

    vocab_index = {km: i for i, km in enumerate(vocab)}
    X = np.zeros((len(sequences), len(vocab)), dtype=np.float32)

    for row, seq in enumerate(sequences):
        counts = kmer_counts(seq, k)
        total = max(sum(counts.values()), 1)
        for km, cnt in counts.items():
            if km in vocab_index:
                X[row, vocab_index[km]] = cnt / total  # relative frequency

    return X, vocab


def gc_content(sequence: str) -> float:
    """Return the GC fraction of `sequence`."""
    if not sequence:
        return 0.0
    gc = sum(1 for nt in sequence.upper() if nt in ("G", "C"))
    return gc / len(sequence)


def build_feature_matrix(
    sequences: list[str],
    k_list: list[int],
    include_gc: bool = True,
    vocab_per_k: dict[int, list[str]] | None = None,
) -> tuple[np.ndarray, dict[int, list[str]]]:
    """
    Stack k-mer frequency vectors (and optionally GC content) into one matrix.

    Returns
    -------
    X : np.ndarray
    vocab_per_k : dict mapping k → vocabulary list  (useful to pass at test time)
    """
    if vocab_per_k is None:
        vocab_per_k = {}

    parts = []
    for k in k_list:
        voc = vocab_per_k.get(k)
        X_k, voc = featurise_kmer(sequences, k, vocab=voc)
        vocab_per_k[k] = voc
        parts.append(X_k)

    if include_gc:
        gc = np.array([gc_content(s) for s in sequences], dtype=np.float32).reshape(-1, 1)
        parts.append(gc)

    return np.hstack(parts), vocab_per_k


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a dict of standard classification metrics."""
    return {
        "accuracy":              round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":              round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "precision_macro":       round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":          round(recall_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "matthews_correlation":  round(matthews_corrcoef(y_true, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Load a sequence-label CSV. Returns (sequences, labels)."""
    df = pd.read_csv(path).dropna(subset=[pd.read_csv(path).columns[0],
                                           pd.read_csv(path).columns[1]])
    df = pd.read_csv(path).dropna()
    sequences = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, -1].astype(int).to_numpy()
    return sequences, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Logistic-regression DNA baseline with k-mer features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_csv",  type=Path, required=True,
                        help="Training data CSV (sequence, label).")
    parser.add_argument("--test_csv",   type=Path, required=True,
                        help="Test data CSV (sequence, label).")
    parser.add_argument("--output_csv", type=Path, required=True,
                        help="Output CSV for results table.")
    parser.add_argument("--kmers", type=str, default="3,4,6",
                        help="Comma-separated k values for k-mer features.")
    parser.add_argument("--no_gc", action="store_true",
                        help="Exclude GC-content feature.")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Max iterations for LogisticRegression solver.")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularisation strength for LogisticRegression.")
    parser.add_argument("--dnabert_results", type=Path, default=None,
                        help="Optional path to DNABERT-2 eval_results.json for side-by-side comparison.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    k_list = [int(k.strip()) for k in args.kmers.split(",")]
    include_gc = not args.no_gc

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    log.info(f"Loading training data from: {args.train_csv}")
    train_seqs, train_labels = load_csv(args.train_csv)
    log.info(f"  {len(train_seqs)} training sequences.")

    log.info(f"Loading test data from: {args.test_csv}")
    test_seqs, test_labels = load_csv(args.test_csv)
    log.info(f"  {len(test_seqs)} test sequences.")

    # -----------------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------------
    log.info(f"Building features — k-mers: {k_list}, GC content: {include_gc}")
    X_train, vocab_per_k = build_feature_matrix(
        train_seqs, k_list, include_gc=include_gc
    )
    X_test, _ = build_feature_matrix(
        test_seqs, k_list, include_gc=include_gc, vocab_per_k=vocab_per_k
    )
    log.info(f"  Feature matrix shape — train: {X_train.shape}, test: {X_test.shape}")

    # -----------------------------------------------------------------------
    # Train & evaluate baselines
    # -----------------------------------------------------------------------
    # We run two variants: with and without feature scaling.
    results = []

    for scale in (False, True):
        label = f"LogReg_{'scaled' if scale else 'raw'}_k{'_'.join(str(k) for k in k_list)}"
        log.info(f"Training: {label} ...")

        steps = []
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append((
            "clf",
            LogisticRegression(
                C=args.C,
                max_iter=args.max_iter,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto",
            ),
        ))
        pipe = Pipeline(steps)
        pipe.fit(X_train, train_labels)

        y_pred = pipe.predict(X_test)
        metrics = evaluate(test_labels, y_pred)
        metrics["model"] = label
        results.append(metrics)

        log.info(
            f"  accuracy={metrics['accuracy']}  "
            f"f1={metrics['f1_macro']}  "
            f"MCC={metrics['matthews_correlation']}"
        )

    # -----------------------------------------------------------------------
    # Optional: load DNABERT-2 results for comparison
    # -----------------------------------------------------------------------
    if args.dnabert_results and args.dnabert_results.exists():
        log.info(f"Loading DNABERT-2 results from: {args.dnabert_results}")
        with open(args.dnabert_results) as f:
            db2 = json.load(f)

        dnabert_row = {
            "model":               "DNABERT-2 (fine-tuned)",
            "accuracy":            db2.get("eval_accuracy", float("nan")),
            "f1_macro":            db2.get("eval_f1", float("nan")),
            "precision_macro":     db2.get("eval_precision", float("nan")),
            "recall_macro":        db2.get("eval_recall", float("nan")),
            "matthews_correlation": db2.get("eval_matthews_correlation", float("nan")),
        }
        results.append(dnabert_row)
    elif args.dnabert_results:
        log.warning(f"DNABERT-2 results file not found: {args.dnabert_results}")

    # -----------------------------------------------------------------------
    # Save & print comparison table
    # -----------------------------------------------------------------------
    col_order = [
        "model", "accuracy", "f1_macro",
        "precision_macro", "recall_macro", "matthews_correlation",
    ]
    results_df = pd.DataFrame(results)[col_order]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    log.info(f"Results table written to: {args.output_csv}")

    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
