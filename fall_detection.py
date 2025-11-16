#!/usr/bin/env python3
"""
fall_check_local.py (refactored v5)

Modes:

1. Motion mode (default):
   - Run motion detection with MOG2.
   - We KEEP the detector timestamps as motion_raw_detect_times_s.
   - We build a window:
         [ first_detected_event - 1s , END_OF_VIDEO ]
     (we always go to the end so we capture final posture).
   - We stratify-sample 6 frames across that entire window
     (2 early / 2 mid / 2 late) and send those to Qwen.

2. Uniform mode (--uniform):
   - Skip motion detection completely.
   - Sample 6 frames at uniform time points over the whole clip
     (1/6 .. 6/6 of total duration).

Output JSON per video now includes:
- motion_raw_detect_times_s: list of timestamps from motion detector
  (or [] in uniform mode, or [mid] if we had to synthesize one).
- frame_sample_times_s: timestamps (sec) of the EXACT frames we sent to Qwen.
  This is what you want to inspect when debugging.
- sample_mode: "motion" or "uniform"

We also still report:
- fall_detected ("yes"/"no"/"unknown")
- final_position ("floor"/"bed"/"standing"/"unknown")
- raw_answer (verbatim model answer)

Other features:
- --debug-save-frames: saves those exact Qwen frames as JPEGs.
- pick_target_device(): fixes 'Invalid device string: "0"' when using device_map="auto".
- detect_shadows arg fixed.
"""

import argparse
import json
import sys
import pathlib
from math import inf
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


######################################
# Motion detection (OpenCV MOG2)
######################################

def detect_motion_events_mog2(
    video_path: str,
    min_area: int = 5000,
    min_motion_time_gap_s: float = 2.0,
    history: int = 200,
    var_threshold: int = 25,
    detect_shadows: bool = True,
) -> Tuple[List[float], float]:
    """
    Returns (event_times, fps)

    event_times = timestamps (sec) where we saw "large motion".
    fps         = frames/sec guess.

    We debounce using min_motion_time_gap_s so if there's continuous motion
    we don't spam events on every frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0  # fallback so math doesn't explode

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )

    last_event_time = -inf
    event_times: List[float] = []

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        ts = frame_idx / fps  # seconds since start

        # Foreground mask
        fgmask = subtractor.apply(frame_bgr)

        # Kill shadows / small flicker
        _, fgmask_bin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        fgmask_clean = cv2.morphologyEx(fgmask_bin, cv2.MORPH_OPEN, kernel)
        fgmask_clean = cv2.morphologyEx(fgmask_clean, cv2.MORPH_DILATE, kernel)

        # Contours => check size
        contours, _ = cv2.findContours(
            fgmask_clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        big_motion = any(cv2.contourArea(cnt) >= min_area for cnt in contours)

        # Debounce / don't fire repeatedly every few ms
        if big_motion and (ts - last_event_time) >= min_motion_time_gap_s:
            event_times.append(ts)
            last_event_time = ts

        frame_idx += 1

    cap.release()
    return event_times, fps


######################################
# Helper: stratified sampling of indices
######################################

def stratified_sample_indices(sorted_indices: List[int], want_total: int) -> List[int]:
    """
    We want ~2 frames from the start, 2 from the middle, 2 from the end
    of the candidate range.

    Split the list into 3 chunks [0:a), [a:b), [b:n).
    Take up to first+last from each chunk.
    """
    if not sorted_indices:
        return []

    n = len(sorted_indices)
    if n <= want_total:
        return sorted_indices

    a = n // 3
    b = (2 * n) // 3

    start_chunk = sorted_indices[0:a] if a > 0 else sorted_indices[0:1]
    mid_chunk   = sorted_indices[a:b] if b > a else sorted_indices[a:a+1]
    end_chunk   = sorted_indices[b:n] if b < n else sorted_indices[-1:]

    def take_two(chunk: List[int]) -> List[int]:
        if len(chunk) <= 2:
            return chunk
        return [chunk[0], chunk[-1]]

    pick_start = take_two(start_chunk)
    pick_mid   = take_two(mid_chunk)
    pick_end   = take_two(end_chunk)

    combined = pick_start + pick_mid + pick_end

    # Deduplicate → sort ascending
    combined = sorted(list(dict.fromkeys(combined)))

    if len(combined) > want_total:
        combined = combined[:want_total]

    return combined


######################################
# Keyframe extraction (motion mode)
######################################

def extract_keyframes_motion_window(
    video_path: str,
    event_times: List[float],
    fps: float,
    pre_first_s: float = 1.0,
    max_frames_total: int = 6,
    resize_to: int = 448,
) -> Tuple[List[Tuple[float, Image.Image]], List[float]]:
    """
    MOTION MODE:

    We *don't* trust last_event anymore.
    We only use motion to anchor the START of "interesting activity".

    Window:
        [ first_event - pre_first_s , END_OF_VIDEO ]

    Then:
    - Convert that window to frame indices.
    - Stratify: 2 early / 2 mid / 2 late.
    - Decode those frames to PIL (resized square).
    - Also return the timestamps (seconds) of those sampled frames.

    Returns:
        frames: list[(t_sec, PIL.Image)]
        frame_times_s: list[float] timestamps of those frames (seconds)
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return [], []

    # If event_times is empty (shouldn't be if we call this, but safety):
    first_ev = min(event_times) if event_times else 0.0

    # Start a bit before first detected motion
    global_start_t = max(first_ev - pre_first_s, 0.0)

    # End of window = full duration of clip
    duration_s = total_frames / fps
    global_end_t = duration_s

    start_idx = int(global_start_t * fps)
    end_idx   = int(global_end_t * fps)

    if end_idx <= start_idx:
        end_idx = start_idx + 1
    if end_idx >= total_frames:
        end_idx = total_frames - 1

    candidate_indices = list(range(start_idx, end_idx + 1))

    sampled_indices = stratified_sample_indices(
        candidate_indices,
        want_total=max_frames_total,
    )

    frames: List[Tuple[float, Image.Image]] = []
    frame_times_s: List[float] = []

    for idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        t_sec = idx / fps
        frame_times_s.append(t_sec)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        if resize_to is not None:
            pil_img = pil_img.resize((resize_to, resize_to))

        frames.append((t_sec, pil_img))

    cap.release()

    # Keep chronological for friendliness
    ordered = sorted(zip(frame_times_s, frames), key=lambda x: x[0])
    frame_times_s = [round(t, 2) for (t, _f) in ordered]
    frames = [f for (_t, f) in ordered]

    return frames, frame_times_s


######################################
# Keyframe extraction (uniform mode)
######################################

def extract_uniform_keyframes(
    video_path: str,
    max_frames_total: int = 6,
    resize_to: int = 448,
) -> Tuple[List[Tuple[float, Image.Image]], List[float], float]:
    """
    UNIFORM MODE:

    No motion detector.
    Take max_frames_total timestamps evenly spaced across the full clip.
    Example: duration=60, N=6 => ~10,20,30,40,50,60 sec.

    Returns:
        frames: list of (t_sec, PIL.Image)
        frame_times_s: list of timestamps (float seconds, rounded 2 decimals)
        fps: fps used (so caller doesn't have to reopen video)
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    if total_frames <= 0:
        cap.release()
        return [], [], fps

    duration_sec = total_frames / fps

    # Choose timestamps at 1/N,2/N,...N/N of total length
    target_times = []
    for i in range(max_frames_total):
        t = (i + 1) * (duration_sec / max_frames_total)
        if t >= duration_sec:
            t = duration_sec
        target_times.append(t)

    sampled_indices = []
    for t in target_times:
        idx = int(t * fps)
        if idx >= total_frames:
            idx = total_frames - 1
        sampled_indices.append(idx)

    # Deduplicate + sort
    sampled_indices = sorted(list(dict.fromkeys(sampled_indices)))

    frames: List[Tuple[float, Image.Image]] = []
    frame_times_s: List[float] = []

    for idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        t_sec = idx / fps
        frame_times_s.append(t_sec)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        if resize_to is not None:
            pil_img = pil_img.resize((resize_to, resize_to))

        frames.append((t_sec, pil_img))

    cap.release()

    # chronological order + rounded times for JSON readability
    ordered = sorted(zip(frame_times_s, frames), key=lambda x: x[0])
    frame_times_s = [round(t, 2) for (t, _f) in ordered]
    frames = [f for (_t, f) in ordered]

    return frames, frame_times_s, fps


######################################
# Debug frame saver
######################################

def save_debug_frames(
    frames: List[Tuple[float, Image.Image]],
    debug_dir: pathlib.Path,
    video_name: str,
):
    """
    Save the exact frames we send to Qwen.
    Each frame:
      <out>/<video_stem>_frames/frame_<idx>_t<timestamp>.jpg
    """
    sub = debug_dir / f"{pathlib.Path(video_name).stem}_frames"
    sub.mkdir(parents=True, exist_ok=True)

    for i, (t_sec, pil_img) in enumerate(frames):
        ts_label = f"{t_sec:.2f}s"
        out_name = f"frame_{i:02d}_t{ts_label}.jpg"
        out_path = sub / out_name
        pil_img.save(out_path, format="JPEG", quality=90)

    print(f"[DEBUG] Saved {len(frames)} Qwen frames under {sub}")


######################################
# Device picker (fixes Invalid device string '0')
######################################

def pick_target_device(model) -> str:
    """
    Return a valid torch device string ('cuda:0' or 'cpu') for inputs.
    Prefer any CUDA shard if available.
    """
    if hasattr(model, "hf_device_map"):
        dev_specs = list(model.hf_device_map.values())

        # Prefer CUDA first
        for d in dev_specs:
            if isinstance(d, int):
                return f"cuda:{d}"
            if isinstance(d, str) and "cuda" in d:
                return d
            if hasattr(d, "type") and d.type == "cuda":
                if d.index is not None:
                    return f"cuda:{d.index}"
                else:
                    return "cuda:0"

        # CPU fallback
        for d in dev_specs:
            if isinstance(d, str) and d.startswith("cpu"):
                return "cpu"
            if hasattr(d, "type") and d.type == "cpu":
                return "cpu"

        return "cpu"

    # Single-device model.device path
    if hasattr(model, "device"):
        md = model.device
        if hasattr(md, "type") and md.type == "cuda":
            idx = md.index if md.index is not None else 0
            return f"cuda:{idx}"
        if hasattr(md, "type") and md.type == "cpu":
            return "cpu"

    return "cpu"


######################################
# Qwen3-VL inference
######################################

def qwen_fall_verdict(
    model,
    processor,
    frames: List[Tuple[float, Image.Image]],
) -> dict:
    """
    Ask Qwen3-VL-2B-Instruct:
    1. Did the person fall to the floor? ('yes'/'no')
    2. Where are they at the end? ('bed'/'floor'/'standing'/'unknown')
    """
    if not frames:
        return {
            "fall_detected": "unknown",
            "final_position": "unknown",
            "raw_answer": "(no frames)"
        }

    # Build multi-modal chat content
    content_list = []
    for (_t_sec, img) in frames:
        content_list.append({
            "type": "image",
            "image": img,
        })

    content_list.append({
        "type": "text",
        "text": (
            "These images show consecutive moments in time. "
            "Question 1: Did the person fall to the floor? "
            "Answer strictly 'yes' or 'no'. "
            "Question 2: After that, where is the person at the end? "
            "Answer with only one word from {bed, floor, standing, unknown}."
        ),
    })

    messages = [
        {
            "role": "user",
            "content": content_list,
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    target_device = pick_target_device(model)

    for k, v in list(inputs.items()):
        if hasattr(v, "to"):
            inputs[k] = v.to(target_device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    # Keep only new tokens (strip prompt)
    trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    answer_list = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    raw_answer = answer_list[0] if answer_list else ""
    txt_lower = raw_answer.lower()

    # Parse fall yes/no
    if "yes" in txt_lower:
        fall_detected = "yes"
    elif "no" in txt_lower:
        fall_detected = "no"
    else:
        fall_detected = "unknown"

    # Parse final position
    final_pos = "unknown"
    for cand in ["floor", "bed", "standing"]:
        if cand in txt_lower:
            final_pos = cand
            break

    return {
        "fall_detected": fall_detected,
        "final_position": final_pos,
        "raw_answer": raw_answer.strip(),
    }


######################################
# Per-video pipeline
######################################

def process_single_video(
    video_path: pathlib.Path,
    model,
    processor,
    out_dir: pathlib.Path,
    min_area: int,
    resize_to: int,
    max_frames_total: int,
    debug_save_frames: bool,
    uniform_mode: bool,
):
    """
    uniform_mode == True:
        - Use extract_uniform_keyframes() across the whole clip.
        - No motion detection.

    uniform_mode == False (motion mode):
        - Run motion detection to get motion_raw_detect_times_s.
        - If empty, synthesize one middle timestamp so we still look at something.
        - Build a window:
              [ first_motion - 1s , END_OF_VIDEO ]
          and stratify 6 frames across that span.

    We then:
        - Optionally dump the frames to disk (--debug-save-frames).
        - Send them to Qwen for fall classification.
        - Save JSON with:
            motion_raw_detect_times_s,
            frame_sample_times_s,
            sample_mode,
            fall_detected,
            final_position,
            raw_answer.
    """

    if uniform_mode:
        # Uniform sampling across entire clip
        frames, frame_times_s, fps = extract_uniform_keyframes(
            str(video_path),
            max_frames_total=max_frames_total,
            resize_to=resize_to,
        )
        motion_times = []   # no detection in this mode
        sample_mode = "uniform"
    else:
        # Motion-based sampling
        motion_times, fps = detect_motion_events_mog2(
            str(video_path),
            min_area=min_area,
            min_motion_time_gap_s=2.0,
            history=200,
            var_threshold=25,
            detect_shadows=True,
        )

        # If detector gave nothing at all, synthesize 1 middle timestamp
        if not motion_times:
            cap_tmp = cv2.VideoCapture(str(video_path))
            total_frames_tmp = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_est = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
            cap_tmp.release()
            if total_frames_tmp > 0:
                mid_t = (total_frames_tmp / fps_est) / 2.0
                motion_times = [mid_t]

        frames, frame_times_s = extract_keyframes_motion_window(
            str(video_path),
            motion_times,
            fps,
            pre_first_s=1.0,
            max_frames_total=max_frames_total,
            resize_to=resize_to,
        )
        sample_mode = "motion"

    # Save debug frames (visual inspection of what Qwen saw)
    if debug_save_frames and frames:
        save_debug_frames(
            frames=frames,
            debug_dir=out_dir,
            video_name=video_path.name,
        )

    # Ask Qwen about fall + final posture
    verdict = qwen_fall_verdict(
        model=model,
        processor=processor,
        frames=frames,
    )

    # Prepare JSON summary
    out_obj = {
        "video": video_path.name,
        "sample_mode": sample_mode,
        "fall_detected": verdict["fall_detected"],
        "final_position": verdict["final_position"],
        "raw_answer": verdict["raw_answer"],
        # motion_raw_detect_times_s is EXACTLY what the motion detector fired on
        # (or the synthetic middle point if it fired on nothing)
        "motion_raw_detect_times_s": [round(t, 2) for t in motion_times],
        # frame_sample_times_s are the timestamps of frames ACTUALLY sent to Qwen
        "frame_sample_times_s": frame_times_s,
    }

    out_path = out_dir / f"{video_path.stem}.json"
    out_path.write_text(
        json.dumps(out_obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[OK] {video_path.name} → {out_path}")
    print(json.dumps(out_obj, indent=2))


######################################
# CLI
######################################

def main():
    parser = argparse.ArgumentParser(
        description="Local fall detection / final position check using Qwen3-VL-2B-Instruct"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Either a single .mp4 file or a folder containing .mp4 files"
    )
    parser.add_argument(
        "--out", default="local_outputs",
        help="Output folder for JSON verdicts (and debug frames if enabled)"
    )
    parser.add_argument(
        "--min-area", type=int, default=5000,
        help="Min moving blob area (px) to count as 'motion event' (ignored in --uniform mode)"
    )
    parser.add_argument(
        "--resize-to", type=int, default=448,
        help="Resize sampled frames sent to Qwen (square side length)"
    )
    parser.add_argument(
        "--max-frames-total", type=int, default=6,
        help="Total frames we ever send to Qwen per clip"
    )
    parser.add_argument(
        "--model-name", default="Qwen/Qwen3-VL-2B-Instruct",
        help="Hugging Face model ID or local path"
    )
    parser.add_argument(
        "--debug-save-frames", action="store_true",
        help="If set, save the exact frames we send to Qwen as JPEGs"
    )
    parser.add_argument(
        "--uniform", action="store_true",
        help="If set, IGNORE motion detection and just sample 6 frames uniformly in time"
    )

    args = parser.parse_args()

    in_path = pathlib.Path(args.input).expanduser().resolve()
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading model:", args.model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",   # can shard across CUDA and CPU
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Gather list of .mp4 videos
    videos: List[pathlib.Path] = []
    if in_path.is_dir():
        for p in sorted(in_path.iterdir()):
            if p.suffix.lower() == ".mp4":
                videos.append(p)
    elif in_path.is_file() and in_path.suffix.lower() == ".mp4":
        videos.append(in_path)
    else:
        print(f"[ERR] Input is neither .mp4 nor directory with .mp4: {in_path}", file=sys.stderr)
        sys.exit(2)

    # Process each video
    for vid in videos:
        try:
            process_single_video(
                video_path=vid,
                model=model,
                processor=processor,
                out_dir=out_dir,
                min_area=args.min_area,
                resize_to=args.resize_to,
                max_frames_total=args.max_frames_total,
                debug_save_frames=args.debug_save_frames,
                uniform_mode=args.uniform,
            )
        except Exception as e:
            print(f"[ERR] {vid.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
