"""
Automated prompt improvement module.

Fetches Cloud Run logs, sends them to Gemini for analysis, and writes an
improved system prompt back to GCS.  The agent reads the prompt from GCS
with a 60-second TTL cache so it picks up improvements automatically.
"""

import logging
import os
import time
from typing import Optional

from .prompts.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_BUCKET: str = os.environ.get("CONFIG_BUCKET", "tripletex-agent-config")
PROMPT_BLOB: str = "system_prompt.txt"
GCS_CACHE_TTL: int = 60  # seconds

GCP_PROJECT: str = "ai-nm26osl-1847"
CLOUD_RUN_SERVICE: str = "tripletex-agent"

# Lines containing any of these keywords are kept during log filtering
LOG_KEYWORDS = [
    "422",
    "404",
    "500",
    "error",
    "Error",
    "systemgenererte",
    "validationMessages",
    "Received task",
    "task_complete",
    "Loop detected",
    "Tool call",
    "Tool result",
]

FILTER_MAX_CHARS: int = 8000

# ---------------------------------------------------------------------------
# In-process cache for the GCS prompt
# ---------------------------------------------------------------------------

_cached_prompt: Optional[str] = None
_cache_fetched_at: float = 0.0


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def load_prompt_from_gcs() -> str:
    """
    Return the system prompt, preferring the GCS version.

    Uses a 60-second in-process TTL cache so that every agent invocation does
    not hit GCS.  Falls back to the static SYSTEM_PROMPT on any error.
    """
    global _cached_prompt, _cache_fetched_at

    now = time.monotonic()
    if _cached_prompt is not None and (now - _cache_fetched_at) < GCS_CACHE_TTL:
        return _cached_prompt

    try:
        from google.cloud import storage  # type: ignore

        client = storage.Client()
        bucket = client.bucket(CONFIG_BUCKET)
        blob = bucket.blob(PROMPT_BLOB)

        if not blob.exists():
            logger.info(
                "GCS prompt blob %s/%s not found — using static prompt",
                CONFIG_BUCKET,
                PROMPT_BLOB,
            )
            return SYSTEM_PROMPT

        text = blob.download_as_text(encoding="utf-8")
        if not text.strip():
            logger.warning("GCS prompt is empty — using static prompt")
            return SYSTEM_PROMPT

        _cached_prompt = text
        _cache_fetched_at = now
        logger.info("Loaded system prompt from GCS (%d chars)", len(text))
        return text

    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load prompt from GCS: %s — using static prompt", exc)
        return SYSTEM_PROMPT


def save_prompt_to_gcs(text: str) -> None:
    """Write *text* as the system prompt blob in GCS."""
    global _cached_prompt, _cache_fetched_at

    try:
        from google.cloud import storage  # type: ignore

        client = storage.Client()
        bucket = client.bucket(CONFIG_BUCKET)
        blob = bucket.blob(PROMPT_BLOB)
        blob.upload_from_string(text, content_type="text/plain; charset=utf-8")
        logger.info("Saved improved prompt to GCS (%d chars)", len(text))

        # Invalidate cache so next agent call picks up the new version
        _cached_prompt = text
        _cache_fetched_at = time.monotonic()

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save prompt to GCS: %s", exc)


# ---------------------------------------------------------------------------
# Log fetching
# ---------------------------------------------------------------------------

def fetch_logs(minutes: int = 5) -> str:
    """
    Fetch Cloud Run structured logs for the last *minutes* minutes.

    Uses the google-cloud-logging Python client (not the gcloud CLI).
    Returns all textPayload lines joined with newlines, or an empty string
    on failure.
    """
    try:
        import datetime

        from google.cloud import logging as gcp_logging  # type: ignore

        client = gcp_logging.Client(project=GCP_PROJECT)

        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            minutes=minutes
        )
        timestamp_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

        filter_str = (
            f'resource.type="cloud_run_revision" '
            f'resource.labels.service_name="{CLOUD_RUN_SERVICE}" '
            f'timestamp>="{timestamp_str}"'
        )

        lines = []
        for entry in client.list_entries(filter_=filter_str, order_by=gcp_logging.ASCENDING):
            payload = entry.payload
            if isinstance(payload, str):
                lines.append(payload)
            elif isinstance(payload, dict):
                # Structured JSON logs — pull out the message field if present
                msg = payload.get("message") or payload.get("msg") or str(payload)
                lines.append(msg)

        result = "\n".join(lines)
        logger.info("Fetched %d log lines (%d chars)", len(lines), len(result))
        return result

    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch Cloud Run logs: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Log filtering
# ---------------------------------------------------------------------------

def filter_logs(logs: str) -> str:
    """
    Keep only lines that contain at least one keyword of interest.
    Caps the result at FILTER_MAX_CHARS characters (truncated from the end
    so the most recent events are preserved).
    """
    kept = [
        line
        for line in logs.splitlines()
        if any(kw in line for kw in LOG_KEYWORDS)
    ]
    result = "\n".join(kept)
    if len(result) > FILTER_MAX_CHARS:
        result = result[-FILTER_MAX_CHARS:]
    return result


# ---------------------------------------------------------------------------
# Gemini improvement cycle
# ---------------------------------------------------------------------------

async def run_improvement() -> None:
    """
    Full async improvement cycle:

    1. Fetch recent Cloud Run logs.
    2. Filter to relevant lines.
    3. Load current system prompt from GCS.
    4. Ask Gemini to produce an improved prompt.
    5. Save the result back to GCS.

    Skips silently if logs are too short to be useful.
    """
    logger.info("Starting improvement cycle")

    raw_logs = fetch_logs(minutes=5)
    filtered = filter_logs(raw_logs)

    if len(filtered) < 100:
        logger.info(
            "Filtered logs too short (%d chars) — skipping improvement", len(filtered)
        )
        return

    current_prompt = load_prompt_from_gcs()

    analysis_prompt = f"""You are an expert at improving AI agent system prompts for a Tripletex accounting automation agent.

Below are recent runtime logs from the agent. Analyse the errors and failures, then return an improved system prompt.

### Recent logs (filtered for relevance):
{filtered}

### Current system prompt:
{current_prompt}

### Instructions:
1. Look at the errors in the logs (422, 404, 500, "systemgenererte", "validationMessages", loop detections).
2. Identify which field names, endpoints, account numbers, or required fields are causing failures.
3. Update the system prompt to fix those issues.
4. Focus on: wrong field names, wrong endpoints, wrong account numbers, missing required fields.
5. Keep all correct information and existing knowledge intact — only fix what the logs show is broken.
6. Return ONLY the complete updated system prompt TEXT — not a Python file, not any explanation, just the raw prompt text that will be used directly as the system instruction."""

    try:
        from google import genai as google_genai
        from google.genai import types as genai_types

        gemini = google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        logger.info("Sending improvement request to Gemini")
        response = gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=[genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=analysis_prompt)])],
            config=genai_types.GenerateContentConfig(temperature=0.2),
        )
        improved_text = response.text.strip()

        if not improved_text:
            logger.warning("Gemini returned empty response — skipping save")
            return

        logger.info("Gemini returned improved prompt (%d chars)", len(improved_text))
        save_prompt_to_gcs(improved_text)

    except Exception as exc:  # noqa: BLE001
        logger.error("Improvement cycle failed during Gemini call: %s", exc)
