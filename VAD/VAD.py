# ---- speech_only_vad.py ----
# Keep only speech (VAD=1) using TenVad; mirror folders; save WAV + PNG (aligned time axes)
import os, math
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa, librosa.display
import matplotlib.pyplot as plt

# If TenVad is in ../include, add it. Otherwise, remove next two lines.
import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../include")))
from ten_vad import TenVad  # expects int16 frames of length hop_size

AUDIO_EXTS = (".wav", ".ogg", ".mp3")

# ---------- resampling without resampy ----------
def _resample_poly_safe(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly
        g = math.gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        y_rs = resample_poly(y, up, down).astype(np.float32, copy=False)
        return y_rs
    except Exception:
        # fallback: simple linear (good enough for VAD)
        n_new = int(round(len(y) * target_sr / sr))
        x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
        return np.interp(x_new, x_old, y).astype(np.float32, copy=False)

# ---------- I/O ----------
def read_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Read wav/ogg/mp3 with librosa (sr=None), mono float32; resample with scipy if needed."""
    y, sr = librosa.load(path, sr=None, mono=True)  # decode at native SR
    y = y.astype(np.float32, copy=False)
    y = _resample_poly_safe(y, sr, target_sr)
    return y, target_sr

# ---------- VAD ----------
def run_ten_vad_flags(audio: np.ndarray, hop_size: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return per-frame (prob, flag) arrays using TenVad.
    TenVad requires int16 frames of exact length hop_size.
    """
    n = len(audio)
    num_frames = int(math.ceil(n / hop_size))
    probs = np.zeros(num_frames, dtype=np.float32)
    flags = np.zeros(num_frames, dtype=np.int16)

    vad = TenVad(hop_size, threshold)
    for i in range(num_frames):
        start = i * hop_size
        end = min((i + 1) * hop_size, n)
        frame_f32 = audio[start:end]

        # pad last frame to hop_size
        if end - start < hop_size:
            pad = np.zeros(hop_size - (end - start), dtype=np.float32)
            frame_f32 = np.concatenate([frame_f32, pad], axis=0)

        # float32 [-1,1] -> int16 for TenVad
        frame_i16 = np.clip(frame_f32 * 32767.0, -32768.0, 32767.0).astype(np.int16)
        p, f = vad.process(frame_i16)   # 1 = speech, 0 = non-speech
        probs[i] = float(p)
        flags[i] = int(f)
    return probs, flags

def flags_to_segments(flags: np.ndarray, hop_size: int, n_samples: int) -> list[tuple[int,int]]:
    """Merge consecutive speech frames into sample-index segments."""
    segs = []
    in_seg = False
    start = 0
    for i, f in enumerate(flags):
        if f == 1 and not in_seg:
            in_seg = True
            start = i * hop_size
        elif f == 0 and in_seg:
            end = min(i * hop_size, n_samples)
            segs.append((start, end))
            in_seg = False
    if in_seg:
        segs.append((start, n_samples))
    return segs

# ---------- plotting ----------
def save_vad_figure(
    full_in_path: str,
    out_png_path: str,
    audio: np.ndarray,
    sr: int,
    flags: np.ndarray,
    hop_size: int
):
    """Top: waveform with VAD shading (speech). Bottom: mel spectrogram. Shared x-axis."""
    dur_s = len(audio) / sr
    t = np.arange(len(audio)) / sr
    dec_samples = np.repeat(flags, hop_size)[: len(audio)].astype(bool)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=300, sharex=True)

    # Waveform + shaded speech
    ymax = float(np.max(np.abs(audio))) if audio.size else 1.0
    ymin, ymax = -max(1e-6, ymax), max(1e-6, ymax)
    ax1.plot(t, audio, lw=0.8, label="Waveform")
    ax1.fill_between(t, ymin, ymax, where=dec_samples, color="tab:red", alpha=0.2, step="pre", label="Speech (VAD)")
    ax1.set_title(f"Waveform with VAD — {full_in_path}")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend(loc="upper right")
    ax1.set_xlim(0, dur_s)

    # Mel spectrogram aligned in time
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=1024, hop_length=hop_size,
        n_mels=128, fmin=20, fmax=min(8000, sr//2), power=2.0, center=False
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_size, x_axis="time", y_axis="mel",
                                   fmin=20, fmax=min(8000, sr//2), ax=ax2)
    cb = fig.colorbar(img, ax=ax2, pad=0.01)
    cb.set_label("dB")
    ax2.set_title("Mel spectrogram")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Mel frequency")
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.set_xlim(0, dur_s)

    plt.tight_layout()
    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png_path, dpi=300)
    plt.close(fig)

# ---------- processing ----------
def process_one_file(
    in_path: str,
    out_wav_path: str,
    out_png_path: str,
    target_sr: int = 16000,
    hop_size: int = 256,
    threshold: float = 0.8
):
    # Read & resample
    audio, sr = read_audio(in_path, target_sr=target_sr)
    orig_len = len(audio)

    # VAD → flags
    probs, flags = run_ten_vad_flags(audio, hop_size=hop_size, threshold=threshold)
    segments = flags_to_segments(flags, hop_size=hop_size, n_samples=len(audio))

    # Concatenate speech segments
    if segments:
        chunks = [audio[s:e] for (s, e) in segments]
        speech = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    else:
        speech = np.array([], dtype=np.float32)

    # Stats
    dur_before = orig_len / sr
    dur_after = len(speech) / sr
    retained_pct = (dur_after / dur_before * 100.0) if dur_before > 0 else 0.0

    # Save audio if any speech
    if speech.size > 0:
        Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_wav_path, speech, sr, subtype="PCM_16", format="WAV")
    else:
        out_wav_path = "(no speech, audio not saved)"

    # Save figure (always, so you can inspect VAD decisions)
    Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    save_vad_figure(in_path, out_png_path, audio, sr, flags, hop_size)

    # Log
    print(
        f"[ok] {in_path}\n"
        f"     before: {dur_before:.2f} s\n"
        f"     after:  {dur_after:.2f} s\n"
        f"     retained (speech): {retained_pct:.2f}%\n"
        f"     -> {out_wav_path}"
    )

def process_tree(
    input_root: str,
    output_root: str,
    figure_root: str,
    target_sr: int = 16000,
    hop_size: int = 256,
    threshold: float = 0.8
):
    input_root  = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)
    figure_root = os.path.abspath(figure_root)

    for dirpath, _, filenames in os.walk(input_root):
        for fn in filenames:
            if not fn.lower().endswith(AUDIO_EXTS):
                continue
            in_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(in_path, input_root)

            out_dir = os.path.join(output_root, os.path.dirname(rel))
            fig_dir = os.path.join(figure_root, os.path.dirname(rel))

            base = os.path.splitext(os.path.basename(rel))[0]
            out_wav_path = os.path.join(out_dir, base + ".wav")  # force WAV output
            out_png_path = os.path.join(fig_dir, base + ".png")

            try:
                process_one_file(
                    in_path,
                    out_wav_path,
                    out_png_path,
                    target_sr=target_sr,
                    hop_size=hop_size,
                    threshold=threshold
                )
            except Exception as e:
                print(f"[error] {in_path}: {e}")

# -------- run inside IDE: set your paths here --------
if __name__ == "__main__":
    input_root  = r"E:\parents"
    output_root = r"E:\parents2"
    figure_root = r"E:\train_audio_figs"
    threshold   = 0.3     # higher → more conservative; lowers false speech
    hop_size    = 256     # TenVad hop; 256 @ 16 kHz ≈ 16 ms
    target_sr   = 16000   # TenVad expects hop_size samples at this SR

    process_tree(input_root, output_root, figure_root,
                 target_sr=target_sr, hop_size=hop_size, threshold=threshold)
