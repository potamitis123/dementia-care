# -*- coding: utf-8 -*-
"""
Sliding-window emotion → AGITATED / NEUTRAL using a categorical SUPERB model
Model: superb/hubert-large-superb-er  (labels: angry, happy, sad, neutral)

- No AutoProcessor (avoids tokenizer/vocab issues). Uses AutoFeatureExtractor.
- Window-level predictions folded to {AGITATED, NEUTRAL}.
- Clip label by fraction of agitated windows (configurable).
- Top subplot: waveform with AGITATED windows shaded; title shows CLIP LABEL.
- Bottom subplot: p(AGITATED) per window (0–1).
- All numbers printed with exactly 3 decimals.

Call from IDE:
    main(data_root="D:/parents/emotions2", window_s=2.0, hop_s=0.5, frac_thr=0.30)
"""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# -------------------------
# Settings
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "superb/hubert-large-superb-er"  # or "superb/wav2vec2-large-superb-er"

# Sliding window (2s/0.5s catches brief shouts)
WINDOW_S = 2.0
HOP_S = 0.5

# Clip-level decision: clip is AGITATED if >= this fraction of windows are agitated
FRAC_AGI_THR = 0.165  # try 0.20–0.40 depending on your tolerance

# Allowed audio extensions
ALLOWED_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

# -------------------------
# Load extractor + model (no tokenizer)
# -------------------------
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
TARGET_SR = int(getattr(feature_extractor, "sampling_rate", 16000))
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(device).eval()

# Label map
ID2LABEL = {int(k): v.lower() for k, v in model.config.id2label.items()}
LABELS = list(ID2LABEL.values())

# Define folding to binary
AGITATED_SET = {"angry", "happy"}
NEUTRAL_SET = {"neutral", "sad"}

def _fmt3(x) -> str:
    return f"{float(x):.3f}"

def _fmt3_list(arr) -> str:
    return "[" + ", ".join(_fmt3(v) for v in arr) + "]"

# -------------------------
# Audio I/O
# -------------------------
def load_audio_any(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return wav.squeeze(0).cpu().numpy().astype(np.float32), target_sr

# -------------------------
# Sliding-window inference (categorical)
# -------------------------
@torch.no_grad()
def predict_windows(path: str, window_s: float = WINDOW_S, hop_s: float = HOP_S) -> Dict:
    wav, _ = load_audio_any(path, TARGET_SR)
    win = int(window_s * TARGET_SR)
    hop = int(hop_s * TARGET_SR)
    n = len(wav)
    if n < win:
        wav = np.pad(wav, (0, win - n))
        n = len(wav)

    probs_list = []
    pred_labels = []
    starts = []

    for start in range(0, max(1, n - win + 1), hop):
        end = start + win
        seg = wav[start:end]

        inputs = feature_extractor(
            seg, sampling_rate=TARGET_SR, return_tensors="pt",
            padding="max_length", truncation=True, max_length=win
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # [num_labels]
        probs_list.append(probs)
        pred_labels.append(ID2LABEL[int(torch.argmax(logits).item())])
        starts.append(start / TARGET_SR)

    probs_arr = np.vstack(probs_list) if probs_list else np.zeros((0, len(LABELS)), dtype=np.float32)

    # Fold to binary: p_agitated = p(angry)+p(happy) if present; else 1 - p(neutral)-p(sad) if available.
    label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}
    p_agitated = np.zeros((probs_arr.shape[0],), dtype=np.float32)
    if AGITATED_SET.issubset(label_to_idx.keys()):
        p_agitated = probs_arr[:, label_to_idx["angry"]] + probs_arr[:, label_to_idx["happy"]]
    else:
        p_neu = probs_arr[:, label_to_idx["neutral"]] if "neutral" in label_to_idx else 0.0
        p_sad = probs_arr[:, label_to_idx["sad"]] if "sad" in label_to_idx else 0.0
        p_agitated = 1.0 - (p_neu + p_sad)
    p_agitated = np.clip(p_agitated, 0.0, 1.0)

    bin_labels = np.where(p_agitated >= 0.5, "AGITATED", "NEUTRAL").tolist()

    return {
        "wav": wav,
        "sr": TARGET_SR,
        "starts_s": np.array(starts, dtype=np.float32),
        "window_s": window_s,
        "hop_s": hop_s,
        "labels_per_window": pred_labels,   # original categorical labels
        "p_agitated": p_agitated,           # folded probability per window
        "probs_raw": probs_arr,             # per-class probabilities
        "class_labels": LABELS,
        "bin_labels": bin_labels,           # AGITATED/NEUTRAL per window
    }

# -------------------------
# Plot (top: waveform shaded; bottom: p(AGITATED))
# -------------------------
def plot_wave_and_prob(base_path: Path, pack: Dict, clip_label: str, save_png=True):
    wav, sr = pack["wav"], pack["sr"]
    t = np.arange(len(wav)) / float(sr)
    starts_s = pack["starts_s"]; win_s = float(pack["window_s"])
    bin_labels = pack["bin_labels"]
    p_ag = pack["p_agitated"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)

    # Top: waveform with AGITATED windows shaded + CLIP LABEL in title
    ax1 = axes[0]
    wmax = np.max(np.abs(wav))
    wdisp = wav / wmax if wmax > 0 else wav
    ax1.plot(t, wdisp, linewidth=0.8)
    for i, lab in enumerate(bin_labels):
        if lab == "AGITATED":
            s = starts_s[i]; e = s + win_s
            ax1.axvspan(s, e, alpha=0.25)
    ax1.set_xlim(0.0, max(t[-1], 1e-6))
    ax1.set_ylabel("Waveform (norm.)")
    ax1.set_title(f"{base_path.name} — CLIP LABEL: {clip_label}")

    # Bottom: p(AGITATED) per window
    ax2 = axes[1]
    x = np.arange(len(p_ag))
    ax2.plot(x, p_ag, label="p(AGITATED)")
    ax2.axhline(0.5, linestyle="--", linewidth=0.8, label="0.5")
    ax2.set_xlabel("Frames (sliding windows)")
    ax2.set_ylabel("Probability (0–1)")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    if save_png:
        out_png = base_path.with_suffix("")
        out_png = Path(str(out_png) + "_er.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return out_png
    else:
        plt.show()
        return None

# -------------------------
# Folder runner
# -------------------------
def run_folder(folder_path: str, window_s: float = WINDOW_S, hop_s: float = HOP_S, frac_thr: float = FRAC_AGI_THR):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
    if not files:
        print(f"No audio files found in: {folder}")
        return

    print(f"Found {len(files)} audio files under {folder}.")
    for f in sorted(files):
        try:
            pack = predict_windows(str(f), window_s=window_s, hop_s=hop_s)

            # Clip decision: fraction of windows with p_agitated >= 0.5
            frac = float(np.mean(pack["p_agitated"] >= 0.5)) if len(pack["p_agitated"]) else 0.0
            clip_label = "AGITATED" if frac >= frac_thr else "NEUTRAL"

            # Print with EXACTLY 3 decimals
            print(f"\n{f}")
            print(f"  windows: {len(pack['p_agitated'])} (win={_fmt3(pack['window_s'])}s, hop={_fmt3(pack['hop_s'])}s)")
            print(f"  raw labels  : {pack['labels_per_window']}")
            print(f"  p(AGITATED) : {_fmt3_list(pack['p_agitated'])}")
            print(f"  bin labels  : {pack['bin_labels']}")
            print(f"  -> CLIP LABEL: {clip_label} (frac_agitated={_fmt3(frac)}, thr={_fmt3(frac_thr)})")

            out_png = plot_wave_and_prob(f, pack, clip_label, save_png=True)
            print(f"  Plot saved to: {out_png}")

        except Exception as e:
            print(f"[ERROR] {f}: {e}")

# -------------------------
# IDE-friendly entry point
# -------------------------
def main(data_root: str,
         window_s: float = WINDOW_S,
         hop_s: float = HOP_S,
         frac_thr: float = FRAC_AGI_THR):
    """
    Example:
        main(data_root="D:/parents/emotions2", window_s=2.0, hop_s=0.5, frac_thr=0.30)
    """
    run_folder(data_root, window_s=window_s, hop_s=hop_s, frac_thr=frac_thr)

# -------------------------
# If running as a script
# -------------------------
if __name__ == "__main__":
    main(data_root="D:/parents/emotions2", window_s=2.0, hop_s=0.5, frac_thr=0.165)