"""
run_motif_injection.py
======================
Motif injection analysis for fine-tuned DNABERT-2 sequence classifiers.

For each DNA sequence in the input CSV, this script:
  1. Runs inference with the original sequence.
  2. Injects each specified motif at a given position (default: center).
  3. Runs inference on the motif-modified sequence.
  4. Reports per-motif label flip rates (0→1 and 1→0).

Outputs
-------
<output_csv>               — per-sequence predictions (original + all motif variants)
<output_csv>_flip_stats.csv — per-motif flip-rate summary table

Usage
-----
    python run_motif_injection.py \\
        --input_csv     data/test.csv \\
        --model_path    output/dnabert2 \\
        --output_csv    results/motif_results.csv \\
        --motifs        ATCG,GCTA,TTAA \\
        --motif_position 25          # optional; defaults to sequence center
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

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
# Motif utilities
# ---------------------------------------------------------------------------
def inject_motif(sequence: str, motif: str, position: int | None = None) -> str:
    """
    Replace a sub-string of `sequence` with `motif` starting at `position`.

    Parameters
    ----------
    sequence : str
        Original DNA sequence (A/T/C/G).
    motif : str
        Short DNA motif to inject.
    position : int or None
        0-based start index. Defaults to the centre of the sequence.

    Returns
    -------
    str
        Modified sequence of the same length as the input.

    Raises
    ------
    ValueError
        If the motif is longer than the sequence or the position is out of bounds.
    """
    seq_len = len(sequence)
    motif_len = len(motif)

    if motif_len > seq_len:
        raise ValueError(
            f"Motif length ({motif_len}) exceeds sequence length ({seq_len})."
        )

    if position is None:
        position = (seq_len - motif_len) // 2

    if position < 0 or position + motif_len > seq_len:
        raise ValueError(
            f"Motif position {position} is out of bounds for a sequence of "
            f"length {seq_len} with motif length {motif_len}."
        )

    return sequence[:position] + motif + sequence[position + motif_len:]


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(
    sequences: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = 32,
) -> list[int]:
    """
    Run batched inference and return the argmax class for each sequence.

    Parameters
    ----------
    sequences : list of str
    tokenizer : HuggingFace tokenizer
    model : HuggingFace classification model (eval mode)
    device : torch.device
    batch_size : int

    Returns
    -------
    list of int
        Predicted class index for each sequence.
    """
    predictions = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        preds = torch.argmax(softmax(logits, dim=-1), dim=-1).cpu().tolist()
        predictions.extend(preds)
    return predictions


# ---------------------------------------------------------------------------
# Flip-rate computation
# ---------------------------------------------------------------------------
def compute_flip_stats(
    original_preds: list[int],
    motif_preds: list[int | None],
    motif: str,
) -> dict:
    """
    Compute label flip counts and rates between original and motif predictions.

    Returns a dict with keys: Motif, Total_1s, Total_0s,
    Flips_1_to_0, Flips_0_to_1, FlipRate_1_to_0, FlipRate_0_to_1.
    """
    flips_1_to_0 = flips_0_to_1 = total_1s = total_0s = 0

    for orig, mod in zip(original_preds, motif_preds):
        if mod is None:
            continue
        if orig == 1:
            total_1s += 1
            if mod == 0:
                flips_1_to_0 += 1
        elif orig == 0:
            total_0s += 1
            if mod == 1:
                flips_0_to_1 += 1

    return {
        "Motif": motif,
        "Total_1s": total_1s,
        "Total_0s": total_0s,
        "Flips_1_to_0": flips_1_to_0,
        "Flips_0_to_1": flips_0_to_1,
        "FlipRate_1_to_0": flips_1_to_0 / total_1s if total_1s > 0 else 0.0,
        "FlipRate_0_to_1": flips_0_to_1 / total_0s if total_0s > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motif injection analysis for DNABERT-2 classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_csv", type=Path, required=True,
        help="CSV with columns [sequence, label] or [sequence1, sequence2, label].",
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to fine-tuned model directory (or HuggingFace model ID).",
    )
    parser.add_argument(
        "--output_csv", type=Path, required=True,
        help="Path for the per-sequence output CSV.",
    )
    parser.add_argument(
        "--motifs", type=str, default=None,
        help="Comma-separated list of DNA motifs to inject (e.g. ATCG,GCTA).",
    )
    parser.add_argument(
        "--motif_position", type=int, default=None,
        help="0-based index at which to inject each motif. Defaults to sequence centre.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Inference batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    log.info(f"Loading input data from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    df = df.dropna(subset=[df.columns[0], df.columns[1]])
    sequences: list[str] = df.iloc[:, 0].tolist()
    true_labels: list = df.iloc[:, 1].tolist()
    log.info(f"  {len(sequences)} sequences loaded.")

    # -----------------------------------------------------------------------
    # Load model & tokenizer
    # -----------------------------------------------------------------------
    log.info(f"Loading model from: {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model.eval().to(device)

    # -----------------------------------------------------------------------
    # Original predictions
    # -----------------------------------------------------------------------
    log.info("Running inference on original sequences...")
    original_preds = predict(sequences, tokenizer, model, device, args.batch_size)

    # -----------------------------------------------------------------------
    # Motif injection
    # -----------------------------------------------------------------------
    motif_list: list[str] = (
        [m.strip() for m in args.motifs.split(",")] if args.motifs else []
    )
    motif_predictions: dict[str, list[int | None]] = {}

    for motif in motif_list:
        log.info(f"Injecting motif: {motif!r}  (position={args.motif_position})")
        modified_seqs: list[str | None] = []
        failed = 0

        for seq in tqdm(sequences, desc=f"  Modifying [{motif}]", leave=False):
            try:
                modified_seqs.append(inject_motif(seq, motif, args.motif_position))
            except ValueError as exc:
                log.warning(f"    Skipping sequence — {exc}")
                modified_seqs.append(None)
                failed += 1

        if failed:
            log.warning(f"  {failed} sequences skipped for motif {motif!r}.")

        # Run inference only on successfully modified sequences
        valid_indices = [i for i, s in enumerate(modified_seqs) if s is not None]
        valid_seqs = [modified_seqs[i] for i in valid_indices]
        valid_preds = predict(valid_seqs, tokenizer, model, device, args.batch_size)

        preds: list[int | None] = [None] * len(sequences)
        for idx, pred in zip(valid_indices, valid_preds):
            preds[idx] = pred
        motif_predictions[motif] = preds

    # -----------------------------------------------------------------------
    # Build output dataframe
    # -----------------------------------------------------------------------
    result_data: dict = {
        "OriginalSequence": sequences,
        "TrueLabel": true_labels,
        "PredictedLabel_Original": original_preds,
    }
    for motif, preds in motif_predictions.items():
        result_data[f"PredictedLabel_Modified_{motif}"] = preds

    result_df = pd.DataFrame(result_data)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_csv, index=False)
    log.info(f"Per-sequence results written to: {args.output_csv}")

    # -----------------------------------------------------------------------
    # Flip statistics
    # -----------------------------------------------------------------------
    if motif_list:
        flip_rows = [
            compute_flip_stats(original_preds, motif_predictions[m], m)
            for m in motif_list
        ]
        flip_df = pd.DataFrame(flip_rows)

        flip_path = args.output_csv.with_name(
            args.output_csv.stem + "_flip_stats.csv"
        )
        flip_df.to_csv(flip_path, index=False)

        log.info(f"Flip statistics written to: {flip_path}")
        print("\n=== Flip Summary ===")
        print(flip_df.to_string(index=False))


if __name__ == "__main__":
    main()
