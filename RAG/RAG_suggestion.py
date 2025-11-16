"""
RAG_suggestion_single.py

End-to-end script to:
1) Index 'Caring_dementia_guide.pdf' into a ChromaDB collection.
2) Parse a daily audio-events JSON file (handles arrays, JSON Lines, or concatenated objects).
3) Retrieve relevant caregiving guidance and generate a personalized caregiver report.

Requirements:
  pip install chromadb pypdf openai pandas

Environment:
  export OPENAI_API_KEY="YOUR_KEY"  # Set your OpenAI API key

Usage:
  python RAG_suggestion.py --json sound_events_single_day.json --pdf Caring_dementia_guide.pdf
"""

import os
import json
import argparse
import re
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from pypdf import PdfReader

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI


# ================================
# Debugging
# ================================
DEBUG = False


def debug_print(*args):
    """Print only when --debug is enabled."""
    if DEBUG:
        print("[DEBUG]", *args)


# ================================
# Configuration
# ================================
DEFAULT_PDF_PATH = "Caring_dementia_guide.pdf"
DEFAULT_JSON_PATH = "sound_events_single_day.json"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "care_dementia_collection"
EMBED_MODEL = "text-embedding-3-small"

# Use environment variable (safer than hard-coding)
OPENAI_KEY = "__API_KEY__"

COUPLE_PROFILE = """\
Female: 87; name variants: Vagelia / Vagelio / Vago / Vangelitsa.
Male: 90+; name variants: Maki / Makis / Mak.
Diagnosis: both have dementia (stage 2); only the female has occasional delusions.
Mobility: both use canes; cannot leave home; cannot access garden (steep staircase); cannot cook or make hot beverages.
Memory: acute short-term memory loss.
Care context: live in own home; two daytime caregivers (Eve, Gogo) on 8-hour shifts for two years; one adult daughter + one other relative visit occasionally; no other visitors.
Residency & property: ~30 years in current home; before that only their youth home (>30 years ago); do not own any other house.
Parents deceased ~50 years ago.
"""

DEFAULT_QUERY = (
    "dementia caregiver best practices: communication, orientation, agitation de-escalation, "
    "hydration, safety, routines, evening confusion (sundowning), cough/health monitoring"
)


# ================================
# 1) Indexing – PDF → Chroma
# ================================
def extract_text_from_pdf(pdf_path: str) -> List[str]:
    debug_print(f"Extracting text from PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text()
        if txt:
            pages.append(txt)
            if i == 1:
                # Show a sample of what will be embedded
                debug_print("Sample text from page 1 (truncated to 500 chars):")
                debug_print(txt[:500] + ("..." if len(txt) > 500 else ""))
    debug_print(f"Extracted text from {len(pages)} pages with content.")
    return pages


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ensure_collection(pdf_path: str,
                      collection_name: str,
                      chroma_dir: str,
                      embed_model: str,
                      api_key: str):
    """Create or open a persistent Chroma collection; index the guide if empty."""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please set it in your environment.")

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embed_model
        ),
    )

    existing_count = collection.count()
    debug_print(f"Chroma collection '{collection_name}' currently has {existing_count} chunks.")

    if existing_count == 0:
        debug_print(f"Indexing {pdf_path} into Chroma…")

        pages = extract_text_from_pdf(pdf_path)

        docs, metas, ids = [], [], []
        uid = 0
        for p, page_text in enumerate(pages, start=1):
            for ch in chunk_text(page_text):
                docs.append(ch)
                metas.append({"page": p})
                ids.append(f"{p}-{uid}")
                uid += 1

        debug_print(f"Prepared {len(docs)} chunks for embedding/indexing.")
        if docs:
            debug_print("First chunk (truncated to 400 chars):")
            debug_print(docs[0][:400] + ("..." if len(docs[0]) > 400 else ""))

        t0 = time.time()
        collection.add(documents=docs, metadatas=metas, ids=ids)
        t1 = time.time()
        debug_print(f"Embedding + adding {len(docs)} chunks took {t1 - t0:.2f} seconds.")
    else:
        debug_print(f"Using existing collection '{collection_name}' with {existing_count} chunks.")

    return collection


# ================================
# 2) Retrieval – query top-k chunks
# ================================
def retrieve_context(collection,
                     query: str,
                     top_k: int = 6,
                     max_chars: int = 4000) -> Tuple[str, List[int]]:
    """Return concatenated context text and list of page numbers for traceability."""
    debug_print(f"Retrieving context for query: {query}")
    t0 = time.time()
    res = collection.query(query_texts=[query], n_results=top_k)
    t1 = time.time()
    debug_print(f"ChromaDB retrieval time: {t1 - t0:.2f} seconds.")

    docs = res["documents"][0] if res and res.get("documents") else []
    metas = res["metadatas"][0] if res and res.get("metadatas") else []

    pages = sorted({m.get("page") for m in metas if isinstance(m, dict) and "page" in m})

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        debug_print(
            f"Result {i}: page={meta.get('page')}, "
            f"chunk_length={len(doc)}"
        )
        debug_print("Chunk preview (first 300 chars):")
        debug_print(doc[:300] + ("..." if len(doc) > 300 else ""))

    ctx = "\n\n".join(docs)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n…"
        debug_print(f"Context truncated to {max_chars} characters.")

    debug_print(f"Total concatenated context length: {len(ctx)} characters.")
    debug_print(f"Pages used for context: {pages}")

    return ctx, pages


# ================================
# 3) IMPROVED JSON parsing
# ================================
def parse_concatenated_json_objects(text: str) -> List[dict]:
    """
    Parse concatenated JSON objects on a single line.
    Handles format: {"eventid": ...}, {"eventid": ...}, {"eventid": ...}
    """
    # Remove BOM and whitespace
    text = text.lstrip("\ufeff").strip()

    # Try standard JSON array first
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass

    # If it's wrapped in array brackets, try fixing
    if text.startswith('[') and text.endswith(']'):
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            text = text[1:-1].strip()  # Remove outer brackets

    # Use brace counting - most reliable method
    objects = parse_by_brace_depth(text)

    if objects:
        debug_print(f"Successfully parsed {len(objects)} objects using brace depth method.")
        return objects

    # Fallback: try wrapping in array
    try:
        wrapped = '[' + text + ']'
        data = json.loads(wrapped)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse JSON. First 200 chars: {text[:200]}")


def parse_by_brace_depth(text: str) -> List[dict]:
    """
    Parse by tracking brace depth, accounting for strings and escapes.
    Most reliable method for concatenated JSON objects.
    """
    items = []
    depth = 0
    start = None
    in_str = False
    esc = False
    i = 0

    while i < len(text):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    obj_str = text[start:i+1]
                    try:
                        obj = json.loads(obj_str)
                        items.append(obj)
                    except json.JSONDecodeError as e:
                        # Debug: print problematic string
                        if len(items) < 5:  # Only print first few errors
                            debug_print(f"Parse error at position {start}: {str(e)[:100]}")
                            debug_print(f"Problematic JSON snippet: {obj_str[:150]}...")
                    start = None

        i += 1

    # Validate we got reasonable results
    if items and all(isinstance(obj, dict) and "eventid" in obj for obj in items):
        return items

    return []


def load_daily_json(json_path: str) -> pd.DataFrame:
    """
    Parse a daily sound-events file where each object contains:
      - eventid, timestamp, audio_events (array), rms, lat, long, related_to

    Returns DataFrame with one row per (event, audio_label).
    """
    print(f"📂 Loading JSON from: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read()

    debug_print("Raw JSON first 300 chars:")
    debug_print(raw[:300] + ("..." if len(raw) > 300 else ""))

    # Parse the concatenated objects
    data = parse_concatenated_json_objects(raw)

    if not data:
        raise ValueError("No valid JSON objects could be parsed from the file.")

    print(f"✅ Successfully parsed {len(data)} event objects")

    # Debug: print first object structure
    if data:
        debug_print(f"Sample object keys: {list(data[0].keys())}")
        if "audio_events" in data[0]:
            debug_print(
                f"Sample audio_events: "
                f"{data[0]['audio_events'][:2] if data[0]['audio_events'] else 'empty'}"
            )

    rows = []
    skipped = 0

    for idx, ev in enumerate(data):
        if not isinstance(ev, dict):
            skipped += 1
            continue

        eventid = ev.get("eventid")
        ts_str = str(ev.get("timestamp", "")).strip()

        # Parse datetime
        dt = pd.to_datetime(ts_str, errors="coerce")
        date_str = dt.strftime("%Y-%m-%d") if pd.notnull(dt) else ""
        time_str = dt.strftime("%H:%M")    if pd.notnull(dt) else ""
        hour_str = dt.strftime("%H")       if pd.notnull(dt) else ""

        rms = pd.to_numeric(ev.get("rms"), errors="coerce")
        lat = pd.to_numeric(ev.get("lat"), errors="coerce")
        lon = pd.to_numeric(ev.get("long"), errors="coerce")
        rel = ev.get("related_to")

        # Explode nested "audio_events"
        audio_events = ev.get("audio_events", [])
        if isinstance(audio_events, list) and audio_events:
            for ae in audio_events:
                if not isinstance(ae, dict):
                    continue
                label = str(ae.get("class", "")).strip().lower()
                prob = pd.to_numeric(ae.get("probability"), errors="coerce")

                if not label:  # Skip empty labels
                    continue

                rows.append({
                    "eventid": eventid,
                    "timestamp": ts_str,
                    "datetime": dt,
                    "date": date_str,
                    "time": time_str,
                    "hour": hour_str,
                    "label": label,
                    "prob": prob,
                    "rms": rms,
                    "lat": lat,
                    "long": lon,
                    "related_to": rel,
                })
        else:
            # If no audio_events, still record the event with no label
            if eventid:  # Only if we have an event ID
                rows.append({
                    "eventid": eventid,
                    "timestamp": ts_str,
                    "datetime": dt,
                    "date": date_str,
                    "time": time_str,
                    "hour": hour_str,
                    "label": "",
                    "prob": np.nan,
                    "rms": rms,
                    "lat": lat,
                    "long": lon,
                    "related_to": rel,
                })

    if not rows:
        raise ValueError(f"No valid data rows extracted from {len(data)} objects. Check JSON structure.")

    df = pd.DataFrame(rows)
    print(f"✅ Created DataFrame with {len(df)} rows from {len(data)} events")
    if skipped:
        print(f"⚠️  Skipped {skipped} malformed objects")

    debug_print("DataFrame head:")
    debug_print(df.head(5).to_string())

    # Normalize columns
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower()
    if "prob" in df.columns:
        df["prob"] = pd.to_numeric(df["prob"], errors="coerce")

    # Sort by datetime - handle missing datetime column
    sort_cols = []
    if "datetime" in df.columns and df["datetime"].notna().any():
        sort_cols.append("datetime")
    if "eventid" in df.columns:
        sort_cols.append("eventid")
    if "label" in df.columns:
        sort_cols.append("label")

    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    return df


# ================================
# 4) JSON summarization
# ================================
def summarize_json_events(df: pd.DataFrame) -> Tuple[str, dict]:
    """Produce a concise textual summary + stats dict."""
    if df.empty:
        return "No events were recorded in the JSON for this day.", {}

    label_stats = (
        df.groupby("label")
          .agg(count=("label", "count"), avg_prob=("prob", "mean"))
          .sort_values("count", ascending=False)
    )

    hour_stats = (
        df.groupby(["hour", "label"])
          .size()
          .reset_index(name="n")
          .sort_values(["hour", "n"], ascending=[True, False])
    )

    distress_keywords = {"whimper", "cry", "scream", "agony", "pain", "moan", "wail", "cough"}
    distress_df = df[df["label"].isin(distress_keywords)]

    top_labels = label_stats.head(6).to_dict(orient="index")

    lines = ["Event summary:"]
    for lbl, row in top_labels.items():
        lines.append(f"- {lbl}: {row['count']} events (avg prob {row['avg_prob']:.2f})")

    if not hour_stats.empty:
        lines.append("\nTime-of-day hotspots (by hour → top label counts):")
        for _, r in hour_stats.head(6).iterrows():
            lines.append(f"- {r['hour']}h: {r['label']} × {int(r['n'])}")

    if not distress_df.empty:
        distress_by_hour = distress_df.groupby("hour").size().sort_values(ascending=False)
        peak_hours = ", ".join([f"{h}h" for h in distress_by_hour.head(3).index.tolist()])
        lines.append(f"\nDistress-like events detected; peak around: {peak_hours}")

    stats_blob = {
        "label_stats": label_stats.reset_index().to_dict(orient="records"),
        "hour_stats": hour_stats.to_dict(orient="records"),
        "distress_labels": sorted(distress_df["label"].unique()) if not distress_df.empty else [],
    }

    summary_text = "\n".join(lines)
    debug_print("JSON summary (first 400 chars):")
    debug_print(summary_text[:400] + ("..." if len(summary_text) > 400 else ""))

    return summary_text, stats_blob


# ================================
# 5) Dynamic retrieval query from JSON
# ================================
def build_retrieval_query_from_json(
    df: pd.DataFrame,
    base_query: str = DEFAULT_QUERY
) -> str:
    """
    Use today's JSON events to specialize the retrieval query.

    Example:
      base_query: generic dementia-care topics
      -> dynamic query mentioning cough, thud, yelling, etc. for today.
    """
    if df.empty or "label" not in df.columns:
        debug_print("JSON is empty or has no 'label' column; using base query only.")
        return base_query

    # Top sound labels for today
    label_counts = (
        df["label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .value_counts()
    )

    # Remove empty labels, keep up to 8
    label_counts = label_counts[label_counts.index != ""].head(8)
    top_labels = label_counts.index.tolist()

    if not top_labels:
        debug_print("No useful labels in JSON; using base query only.")
        return base_query

    top_labels_str = ", ".join(top_labels)

    # Distress / interaction-related sounds get extra emphasis
    distress_keywords = {
        "whimper", "cry", "scream", "agony", "pain", "moan",
        "wail", "cough", "yell", "shout"
    }
    distress_present = [lbl for lbl in top_labels if lbl in distress_keywords]
    distress_str = ", ".join(distress_present) if distress_present else ""

    parts = [base_query]

    # Add a description of today's soundscape
    parts.append(
        "Today's home audio monitoring detected frequent events: "
        f"{top_labels_str}. "
        "Retrieve practical dementia home-care guidance for handling these sounds."
    )

    # If distress sounds appear, highlight them
    if distress_present:
        parts.append(
            "Give extra emphasis to what caregivers should do when these occur: "
            f"{distress_str}."
        )

    # Optionally use hour hotspots for context
    if "hour" in df.columns:
        hot_hours = (
            df[df["hour"].notna()]
            .groupby("hour")["eventid"]
            .count()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        if hot_hours:
            pretty_hours = ", ".join(f"{h}h" for h in hot_hours)
            parts.append(
                f"Consider that events cluster around these hours: {pretty_hours}."
            )

    dynamic_query = " ".join(parts)
    debug_print("Dynamic retrieval query built from JSON:")
    debug_print(dynamic_query)

    return dynamic_query


# ================================
# 6) Generation – LLM prompt
# ================================
def build_prompt(couple_profile: str,
                 context_text: str,
                 context_pages: List[int],
                 json_summary_text: str,
                 raw_json: pd.DataFrame) -> str:
    pages_str = ", ".join(map(str, context_pages)) if context_pages else "N/A"
    preview_rows = min(len(raw_json), 40)

    # Convert DataFrame to dict and handle datetime serialization
    raw_preview = raw_json.head(preview_rows).copy()

    # Convert datetime columns to strings for JSON serialization
    for col in raw_preview.columns:
        if pd.api.types.is_datetime64_any_dtype(raw_preview[col]):
            raw_preview[col] = raw_preview[col].astype(str)

    raw_preview_dict = raw_preview.to_dict(orient="records")

    prompt = f"""
You are a dementia-care assistant analyzing today's audio monitoring data.

COUPLE BACKGROUND:
{couple_profile}

RETRIEVED GUIDANCE (from pages: {pages_str}):
{context_text}

TODAY'S SUMMARY:
{json_summary_text}

RAW DATA (first {preview_rows} rows):
{json.dumps(raw_preview_dict, indent=2, ensure_ascii=False)}

Create a caregiver report with these sections:
1) Summary of the day
2) Behavioral interpretation
3) Caregiver action items (specific, practical)
4) Environment/routine adjustments
5) Safety flags
6) What to monitor tomorrow
7) Evidence used (cite pages)

Keep under 400 words. Be specific and practical.
"""
    debug_print("Prompt length (characters):", len(prompt))
    return prompt


def generate_caregiver_report(json_path: str,
                              retrieval_query: str = DEFAULT_QUERY,
                              pdf_path: str = DEFAULT_PDF_PATH) -> str:
    pipeline_start = time.time()

    # 1) Ensure index/collection (PDF + embeddings)
    collection = ensure_collection(
        pdf_path=pdf_path,
        collection_name=COLLECTION_NAME,
        chroma_dir=CHROMA_DIR,
        embed_model=EMBED_MODEL,
        api_key=OPENAI_KEY,
    )

    # 2) Load & summarize JSON (daily, changes every day)
    df = load_daily_json(json_path)
    json_summary_text, _stats = summarize_json_events(df)

    # 3) Build retrieval query that depends on today's events
    dynamic_query = build_retrieval_query_from_json(
        df=df,
        base_query=retrieval_query
    )

    # 4) Retrieve guidance from PDF using the dynamic query
    context_text, context_pages = retrieve_context(
        collection,
        dynamic_query,
        top_k=6
    )

    # 5) Build prompt (history + context + summary + raw data)
    prompt = build_prompt(
        couple_profile=COUPLE_PROFILE,
        context_text=context_text,
        context_pages=context_pages,
        json_summary_text=json_summary_text,
        raw_json=df,
    )

    debug_print("Final prompt to LLM (full):")
    debug_print(prompt)

    # 6) Call LLM
    client = OpenAI(api_key=OPENAI_KEY)
    t0 = time.time()
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    t1 = time.time()
    debug_print(f"LLM response time: {t1 - t0:.2f} seconds.")

    report = resp.choices[0].message.content.strip()
    debug_print(f"Total pipeline time: {time.time() - pipeline_start:.2f} seconds.")

    return report


# ================================
# CLI
# ================================
def main():
    parser = argparse.ArgumentParser(description="Generate caregiver report using RAG + daily JSON audio.")
    parser.add_argument("--json", type=str, default=DEFAULT_JSON_PATH, help="Path to daily JSON file.")
    parser.add_argument("--pdf", type=str, default=DEFAULT_PDF_PATH, help="Path to dementia guide PDF.")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Retrieval query.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debugging output.")
    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug
    if DEBUG:
        print("🔧 DEBUG MODE ENABLED")

    report = generate_caregiver_report(
        json_path=args.json,
        retrieval_query=args.query,
        pdf_path=args.pdf
    )
    print("\n🩺 CAREGIVER REPORT\n")
    print(report)


if __name__ == "__main__":
    main()
