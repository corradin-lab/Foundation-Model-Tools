"""
log_eval_results.py
===================
Append DNABERT-2 evaluation metrics from a JSON results file to a
running CSV summary table.

This script is designed to be called automatically at the end of a
SLURM fine-tuning job (see slurm_finetune.sh), but can also be run
manually after training completes.

The summary CSV grows one row per run, making it easy to track
experiments over time.

Usage
-----
    python log_eval_results.py \\
        --json_path  output/dnabert2/results/DNABERT2_run1/eval_results.json \\
        --csv_path   SUMMARY_STATS.csv \\
        --run_name   DNABERT2_run1          # optional, inferred from path if omitted
        --extra      "epochs=25,lr=3e-5"   # optional key=value annotations
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Metrics to extract (in order they appear in the output CSV).
METRIC_KEYS = [
    "eval_accuracy",
    "eval_precision",
    "eval_recall",
    "eval_f1",
    "eval_matthews_correlation",
    "eval_loss",
    "eval_runtime",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_eval_results(json_path: Path) -> dict:
    """Load and return the JSON results dict, raising on missing file."""
    if not json_path.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {json_path}\n"
            "Ensure training completed successfully and --output_dir is correct."
        )
    with open(json_path) as f:
        return json.load(f)


def build_row(
    results: dict,
    run_name: str,
    extra: dict[str, str] | None,
) -> dict:
    """Assemble a single CSV row from the results dict."""
    row: dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name":  run_name,
    }
    for key in METRIC_KEYS:
        row[key] = results.get(key, "")
    if extra:
        row.update(extra)
    return row


def append_to_csv(csv_path: Path, row: dict) -> None:
    """
    Append `row` to `csv_path` as a new line.
    Creates the file with a header if it does not already exist.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
            log.info(f"Created summary CSV with header: {csv_path}")
        writer.writerow(row)
        log.info(f"Appended row for run '{row['run_name']}' to: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append DNABERT-2 eval metrics to a summary CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json_path", type=Path, required=True,
        help="Path to eval_results.json produced by train.py.",
    )
    parser.add_argument(
        "--csv_path", type=Path, default=Path("SUMMARY_STATS.csv"),
        help="Path to the running summary CSV (created if absent).",
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Experiment label. Defaults to the parent directory name of --json_path.",
    )
    parser.add_argument(
        "--extra", type=str, default=None,
        help="Comma-separated key=value pairs to annotate the row, e.g. 'epochs=25,lr=3e-5'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Infer run name from the results directory if not provided
    run_name = args.run_name or args.json_path.parent.name

    # Parse optional extra annotations
    extra: dict[str, str] | None = None
    if args.extra:
        try:
            extra = dict(item.split("=", 1) for item in args.extra.split(","))
        except ValueError:
            log.warning(
                f"Could not parse --extra '{args.extra}'. "
                "Expected format: key1=val1,key2=val2"
            )

    # Load results and write to CSV
    log.info(f"Loading evaluation results from: {args.json_path}")
    results = load_eval_results(args.json_path)
    log.info(f"  Keys found: {list(results.keys())}")

    row = build_row(results, run_name, extra)
    append_to_csv(args.csv_path, row)

    # Print a quick summary to stdout
    print("\n=== Logged Metrics ===")
    for k, v in row.items():
        print(f"  {k:<30} {v}")


if __name__ == "__main__":
    main()
