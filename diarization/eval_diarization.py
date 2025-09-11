# -*- coding: utf-8 -*-
"""
Evaluate speaker-set predictions from a CSV (Spyder/IDE-friendly).

CSV columns:
  - file: full path + filename (kept for reference)
  - speakers_detected: predicted label (e.g., 'A', 'A+B', 'A+B+C')
  - truth  (or 'speakers_truth'): ground-truth label (same format)
"""

from __future__ import annotations
from pathlib import Path
from typing import Set, Tuple, Dict, Optional
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,        # subset accuracy in multilabel mode
    hamming_loss,
    jaccard_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ----------------------------
# Helpers
# ----------------------------
def _canon_label(label: str, allowed: Set[str]) -> str:
    """Normalize composite label: split '+', uppercase, dedup, sort, re-join."""
    if label is None:
        return ""
    parts = [p.strip().upper() for p in str(label).split("+") if p.strip()]
    parts = [p for p in parts if p in allowed]
    return "+".join(sorted(set(parts)))

def _to_set(label: str, allowed: Set[str]) -> Set[str]:
    lbl = _canon_label(label, allowed)
    return set(lbl.split("+")) if lbl else set()

def evaluate_from_csv(
    csv_path: str | Path,
    out_dir: str | Path,
    allowed_speakers: Optional[Set[str]] = None,
    pred_col: str = "speakers_detected",
    truth_cols: Tuple[str, ...] = ("truth", "speakers_truth"),
    save_artifacts: bool = True,
    verbose: bool = True,
):
    """
    Evaluate multi-speaker predictions from a CSV.

    Returns:
        per_file_df : DataFrame [file, true_label, pred_label, exact_match]
        combo_cm_df : Confusion matrix over combination labels (e.g., 'A+B')
        metrics     : Dict of scalar metrics
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if allowed_speakers is None:
        allowed_speakers = {"A", "B", "C", "D", "E", "F"}

    df = pd.read_csv(csv_path, delimiter=';')

    # Resolve columns
    if "file" not in df.columns:
        raise ValueError("CSV must contain a 'file' column.")
    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}'.")
    truth_col = next((c for c in truth_cols if c in df.columns), None)
    if truth_col is None:
        raise ValueError(f"CSV must have one of the truth columns: {truth_cols}")

    # Canonicalize labels and compute exact match
    df["true_label"] = df[truth_col].apply(lambda s: _canon_label(s, allowed_speakers))
    df["pred_label"] = df[pred_col].apply(lambda s: _canon_label(s, allowed_speakers))
    df["exact_match"] = (df["true_label"] == df["pred_label"]).astype(int)

    # Multilabel binarization (per-speaker view)
    classes = sorted(allowed_speakers)
    mlb = MultiLabelBinarizer(classes=classes)
    Y_true = mlb.fit_transform(df["true_label"].apply(lambda s: _to_set(s, allowed_speakers)))
    Y_pred = mlb.transform(df["pred_label"].apply(lambda s: _to_set(s, allowed_speakers)))

    # Scalar metrics
    metrics: Dict[str, float] = {
        "subset_acc":     float(accuracy_score(Y_true, Y_pred)),                      # exact set match
        "hamming_loss":   float(hamming_loss(Y_true, Y_pred)),
        "jaccard_samples":float(jaccard_score(Y_true, Y_pred, average="samples")),
        "jaccard_micro":  float(jaccard_score(Y_true, Y_pred, average="micro")),
        "jaccard_macro":  float(jaccard_score(Y_true, Y_pred, average="macro")),
        "f1_samples":     float(f1_score(Y_true, Y_pred, average="samples", zero_division=0)),
        "f1_micro":       float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "f1_macro":       float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
    }

    if verbose:
        print("\n=== MULTILABEL (per-speaker) METRICS ===")
        for k in ["subset_acc","hamming_loss","jaccard_samples","jaccard_micro","jaccard_macro",
                  "f1_samples","f1_micro","f1_macro"]:
            print(f"{k:>16}: {metrics[k]:.4f}")
        print("\nPer-speaker classification report:")
        print(classification_report(Y_true, Y_pred, target_names=classes, zero_division=0))

    # Combination-level confusion matrix (treat 'A+B' as one class)
    comb_true = df["true_label"]
    comb_pred = df["pred_label"]
    combo_labels = sorted(sorted(set(comb_true.unique()) | set(comb_pred.unique())),
                         key=lambda s: (s.count("+"), s))
    cm = confusion_matrix(comb_true, comb_pred, labels=combo_labels)
    combo_cm_df = pd.DataFrame(cm, index=[f"T:{l}" for l in combo_labels],
                                  columns=[f"P:{l}" for l in combo_labels])

    per_file_df = df[["file", "true_label", "pred_label", "exact_match"]].copy()

    if save_artifacts:
        per_file_df.to_csv(out_dir / "speaker_eval_from_csv.csv", index=False)
        combo_cm_df.to_csv(out_dir / "speaker_combo_confusion_from_csv.csv", index=False)
        if verbose:
            print("\nSaved:")
            print(f"- Per-file results: {out_dir / 'speaker_eval_from_csv.csv'}")
            print(f"- Combination confusion matrix: {out_dir / 'speaker_combo_confusion_from_csv.csv'}")

    return per_file_df, combo_cm_df, metrics

# ----------------------------
# Spyder-friendly entry point (edit paths here, then run)
# ----------------------------
def main():
    CSV_PATH = r"D:\parents\file_summary2.csv"   # <- change
    OUT_DIR  = r"D:\parents\evaluation"          # <- change
    ALLOWED  = {"A","B","C","D","E","F"}         # <- adjust if you add speakers

    per_file_df, combo_cm_df, metrics = evaluate_from_csv(
        csv_path=CSV_PATH,
        out_dir=OUT_DIR,
        allowed_speakers=ALLOWED,
        save_artifacts=True,
        verbose=True,
    )

    # Quick peek in console (optional)
    print("\nHead of per-file results:")
    print(per_file_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
