# -*- coding: utf-8 -*-

# ---------- silence warnings/log noise (must be first) ----------
import os, warnings, logging
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
for n in ["pyannote", "pyannote.audio", "urllib3"]:
    logging.getLogger(n).setLevel(logging.ERROR)
# ---------------------------------------------------------------

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans

# Diarization (requires HF token + model access)
from pyannote.audio import Pipeline
# Few-shot speaker embeddings
from speechbrain.inference.speaker import EncoderClassifier

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_ROOT = Path(r"D:\parents\speakers")     # training root (folders like A, B, A+B, A+B+C ...)
TEST_ROOT  = Path(r"D:\parents\test_diarization")    # unknown folder with .ogg/.wav/.mp3 files
OUTPUT_DIR = Path(r"D:\parents")              # where CSVs will be written

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or ""   # paste your token if you prefer: "hf_xxx"

TARGET_SR = 16000
MIN_SEG_DUR = 1.2            # ignore very short segments; increase to 1.5 if needed
MAX_TEST_SPEAKERS = 3        # constrain diarization at test time
FORCE_CHOICE = True          # <-- always assign to the nearest enrolled speaker (no "unknown")
ALLOWED_EXTS = {".ogg", ".wav", ".flac", ".mp3", ".m4a"}

# Devices:
DIAR_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SB_DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# FFmpeg-only loader (bypasses mpg123/libsndfile)
# -----------------------------
def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise RuntimeError("FFmpeg not found. Install it and ensure 'ffmpeg' is on PATH.")

def ffmpeg_decode_mono(path: str, sr: int = TARGET_SR) -> torch.Tensor:
    """
    Decode using FFmpeg CLI only. Returns mono [1, T] float32 in [-1, 1].
    """
    cmd = ["ffmpeg", "-v", "error", "-nostdin",
           "-i", str(path),
           "-ac", "1", "-ar", str(sr),
           "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
    out = subprocess.check_output(cmd)  # raises on hard failures
    y = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(y).unsqueeze(0)  # [1, T]

def load_mono(path: Path, target_sr: int = TARGET_SR) -> Tuple[torch.Tensor, int]:
    wav = ffmpeg_decode_mono(str(path), sr=target_sr)
    return wav, target_sr

# -----------------------------
# MODELS
# -----------------------------
def load_models():
    if not HF_TOKEN:
        print("[INFO] Using HUGGINGFACE_TOKEN from environment (if set).")
    diar = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN if HF_TOKEN else None
    )
    diar.to(DIAR_DEVICE)

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": SB_DEVICE}
    )
    enc.eval()
    return diar, enc

# -----------------------------
# UTILITIES
# -----------------------------
def list_audio_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]

def parse_speakers_from_folder(folder_name: str) -> List[str]:
    """Split on '+' and strip spaces. E.g., 'A+B' -> ['A','B'] (order not meaningful)."""
    return [tok.strip() for tok in folder_name.split("+") if tok.strip()]

@torch.no_grad()
def embed_signal(enc: EncoderClassifier, wav_1xT: torch.Tensor) -> torch.Tensor:
    """Return L2-normalized embedding [1, D]."""
    emb = enc.encode_batch(wav_1xT.to(SB_DEVICE))  # [1, 1, D] or [1, D]
    emb = emb.squeeze()
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
    return emb.cpu()

def assign_segment(emb: torch.Tensor, centroids: Dict[str, torch.Tensor]) -> Tuple[str, float]:
    """Always choose best match (closed-set)."""
    best_name, best_score = None, -1.0
    for name, cen in centroids.items():
        score = F.cosine_similarity(emb, cen).item()
        if score > best_score:
            best_name, best_score = name, score
    # FORCE_CHOICE implies we never return "unknown"
    return best_name, best_score

# -----------------------------
# BOOTSTRAP from multi-speaker folders (if no single-speaker data for some names)
# -----------------------------
def bootstrap_from_multispeaker(
    diar, enc, d: Path, existing_centroids: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Diarize one multi-speaker folder 'd' with num_speakers=K,
    embed all segments, cluster into K groups (K = #names in folder),
    map clusters to sorted names. Seeds centroids only for names not yet enrolled.
    """
    speakers = parse_speakers_from_folder(d.name)
    need = [s for s in speakers if s not in existing_centroids]
    if not need:
        return {}

    K = len(speakers)
    files = list_audio_files(d)
    embs = []
    for f in files:
        try:
            wav, sr = load_mono(f)
            ann = diar({"waveform": wav, "sample_rate": sr}, num_speakers=K)
            for segment, _, _ in ann.itertracks(yield_label=True):
                start, end = float(segment.start), float(segment.end)
                if end - start < MIN_SEG_DUR:
                    continue
                s0, s1 = int(start * sr), int(end * sr)
                seg = wav[:, s0:s1]
                emb = embed_signal(enc, seg)  # [1, D]
                embs.append(emb.squeeze(0).numpy())
        except Exception as e:
            print(f"[WARN] bootstrap {f}: {e}")

    if len(embs) < len(need):
        print(f"[WARN] Not enough segments to bootstrap {d.name}: {len(embs)} found for {len(need)} speakers.")
        return {}

    X = np.stack(embs, axis=0)  # [N, D]
    km = KMeans(n_clusters=len(need), n_init=10, random_state=0)
    labels = km.fit_predict(X)

    centroids = {}
    for k, name in enumerate(sorted(need)):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        mean = torch.tensor(X[idx]).mean(dim=0, keepdim=True)
        mean = mean / (mean.norm(dim=-1, keepdim=True) + 1e-9)
        centroids[name] = mean.cpu()
    return centroids

# -----------------------------
# ENROLLMENT (build centroids)
# -----------------------------
def enroll_from_training(diar, enc, train_root: Path) -> Dict[str, torch.Tensor]:
    """
    Build speaker centroids:
      - For single-speaker folders (K=1): diarize with num_speakers=1, embed, average.
      - For multi-speaker folders (K>=2):
          * if some names are missing centroids, bootstrap them via KMeans;
          * refine centroids by assigning each segment to the nearest name AMONG that folder's names.
    """
    if not train_root.exists():
        raise FileNotFoundError(f"Training root not found: {train_root}")

    subdirs = [d for d in train_root.iterdir() if d.is_dir()]
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    def add_embedding(name: str, emb: torch.Tensor):
        if name not in sums:
            sums[name] = emb.clone()
            counts[name] = 1
        else:
            sums[name] += emb
            counts[name] += 1

    # Pass 1: single-speaker folders (K=1)
    for d in sorted(subdirs):
        speakers = parse_speakers_from_folder(d.name)
        if len(speakers) == 1:
            spk = speakers[0]
            files = list_audio_files(d)
            for f in tqdm(files, desc=f"[enroll-1] {spk}", ncols=80):
                try:
                    wav, sr = load_mono(f)
                    ann = diar({"waveform": wav, "sample_rate": sr}, num_speakers=1)
                    for segment, _, _ in ann.itertracks(yield_label=True):
                        start, end = float(segment.start), float(segment.end)
                        if end - start < MIN_SEG_DUR:
                            continue
                        s0, s1 = int(start * sr), int(end * sr)
                        seg = wav[:, s0:s1]
                        emb = embed_signal(enc, seg)
                        add_embedding(spk, emb)
                except Exception as e:
                    print(f"[WARN] enroll-1 {f}: {e}")

    # Initial centroids
    centroids: Dict[str, torch.Tensor] = {}
    for spk in sums:
        M = sums[spk] / counts[spk]
        M = M / (M.norm(dim=-1, keepdim=True) + 1e-9)
        centroids[spk] = M.cpu()
    if centroids:
        print(f"[enroll-1] Initialized (single-speaker): {sorted(list(centroids.keys()))}")
    else:
        print("[enroll-1] No single-speaker folders found.")

    # Pass 2: multi-speaker folders (K>=2): bootstrap missing names then refine
    for d in sorted(subdirs):
        speakers = parse_speakers_from_folder(d.name)
        K = len(speakers)
        if K >= 2:
            # Bootstrap missing names (if any)
            missing = [s for s in speakers if s not in centroids]
            if missing:
                seeded = bootstrap_from_multispeaker(diar, enc, d, centroids)
                if seeded:
                    centroids.update(seeded)
                    print(f"[bootstrap] Seeded from {d.name}: {sorted(seeded.keys())}")
                else:
                    print(f"[WARN] Could not bootstrap missing speakers from {d.name}: {missing}")

            files = list_audio_files(d)
            for f in tqdm(files, desc=f"[enroll-2] {d.name}", ncols=80):
                try:
                    wav, sr = load_mono(f)
                    ann = diar({"waveform": wav, "sample_rate": sr}, num_speakers=K)
                    for segment, _, _ in ann.itertracks(yield_label=True):
                        start, end = float(segment.start), float(segment.end)
                        if end - start < MIN_SEG_DUR:
                            continue
                        s0, s1 = int(start * sr), int(end * sr)
                        seg = wav[:, s0:s1]
                        emb = embed_signal(enc, seg)

                        # Assign ONLY among names listed in this folder
                        best_name, best_score = None, -1.0
                        for name in speakers:
                            if name not in centroids:
                                continue
                            score = F.cosine_similarity(emb, centroids[name]).item()
                            if score > best_score:
                                best_name, best_score = name, score
                        if best_name is not None:
                            prev_n = counts.get(best_name, 0)
                            new_centroid = (centroids[best_name] * prev_n + emb) / (prev_n + 1)
                            centroids[best_name] = new_centroid / (new_centroid.norm(dim=-1, keepdim=True) + 1e-9)
                            counts[best_name] = prev_n + 1
                except Exception as e:
                    print(f"[WARN] enroll-2 {f}: {e}")

    if not centroids:
        raise RuntimeError("No centroids were created. Check your training folders and audio files.")
    print("[enroll] Final speakers:", ", ".join(sorted(centroids.keys())))
    return centroids

# -----------------------------
# INFERENCE on unknown folder
# -----------------------------
def recognize_folder(
    diar, enc, centroids: Dict[str, torch.Tensor],
    test_root: Path, min_seg_dur: float = MIN_SEG_DUR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Diarize each file in test_root with min/max speaker constraints
    and assign each diarized track to the nearest enrolled speaker (forced choice).
    Track-level pooling (mean of segment embeddings) improves stability.
    """
    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    files = list_audio_files(test_root)
    all_rows = []

    for f in tqdm(files, desc="[infer] files", ncols=80):
        try:
            wav, sr = load_mono(f)
            ann = diar({"waveform": wav, "sample_rate": sr},
                       min_speakers=1, max_speakers=MAX_TEST_SPEAKERS)

            # Collect segments per diarized label (e.g., SPEAKER_00, SPEAKER_01)
            tracks: Dict[str, List[Tuple[float, float]]] = {}
            for segment, _, label in ann.itertracks(yield_label=True):
                start, end = float(segment.start), float(segment.end)
                if end - start >= min_seg_dur:
                    tracks.setdefault(label, []).append((start, end))

            # For each track: average embeddings over its spans, assign once, then propagate
            for label, spans in tracks.items():
                embs = []
                for start, end in spans:
                    s0, s1 = int(start * sr), int(end * sr)
                    seg = wav[:, s0:s1]
                    embs.append(embed_signal(enc, seg))
                speaker_emb = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)
                speaker_emb = speaker_emb / (speaker_emb.norm(dim=-1, keepdim=True) + 1e-9)

                spk_name, spk_score = assign_segment(speaker_emb, centroids)

                for start, end in spans:
                    all_rows.append({
                        "file": str(f),
                        #"track": label,
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "duration": round(end - start, 2),
                        "speaker": spk_name,         # forced assignment
                        "score": round(spk_score, 3),
                    })

        except Exception as e:
            print(f"[WARN] inference {f}: {e}")

    seg_df = pd.DataFrame(all_rows).sort_values(["file", "start"]).reset_index(drop=True)

    # per-file summary (unique speakers found)
    summaries = []
    for file, grp in seg_df.groupby("file"):
        speakers = sorted(grp["speaker"].unique().tolist())
        summaries.append({"file": file, "speakers_detected": "+".join(speakers)})
    summary_df = pd.DataFrame(summaries).sort_values("file").reset_index(drop=True)

    return seg_df, summary_df

# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_ffmpeg()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diar, enc = load_models()

    # 1) Enroll from training data (uses folder names to set num_speakers)
    centroids = enroll_from_training(diar, enc, TRAIN_ROOT)
    torch.save({k: v.numpy() for k, v in centroids.items()}, OUTPUT_DIR / "centroids.pt")

    # Load the model
    #centroids_np = torch.load(OUTPUT_DIR / "centroids.pt", map_location="cpu", weights_only=False)
    #device = next(enc.parameters()).device if any(p.requires_grad for p in enc.parameters()) else torch.device("cpu")
    #centroids = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.as_tensor(v)).to(device, dtype=torch.float32) for k, v in centroids_np.items}

    # 2) Run inference on unknown folder (min=1, max=3 speakers)
    seg_df, summary_df = recognize_folder(diar, enc, centroids, TEST_ROOT)

    # 3) Save outputs
    seg_df.to_csv(OUTPUT_DIR / "segments.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "file_summary.csv", index=False)

    print("\n=== Done ===")
    print(f"Segments CSV:   {OUTPUT_DIR / 'segments.csv'}")
    print(f"Summary CSV:    {OUTPUT_DIR / 'file_summary.csv'}")
    print(f"Centroids file: {OUTPUT_DIR / 'centroids.pt'}")

if __name__ == "__main__":
    main()
