"""LLM narrative synthesis, per PRD: 'orkestrasi LLM untuk menerjemahkan
output probabilitas numerik menjadi narasi kebijakan mitigasi.'

Runs as part of the batch pipeline now (not a live per-request endpoint --
see our earlier architecture-update discussion on why the FastAPI live
router from the previous design no longer fits). Same fixed-template and
error-handling pattern as before -- reused deliberately, this part of the
design didn't actually change, just where it runs."""

import json
import httpx
from src.config import settings

PROMPT_TEMPLATE = """You are a spatial policy analyst for a TOD commercial resilience platform. \
Given the following grid-level data, write a concise 2-3 sentence narrative summary in Bahasa \
Indonesia, then classify into exactly one recommendation_type: "mitigasi", "realokasi", or "pemantauan".

Grid data (JSON):
{payload}

Respond ONLY as JSON: {{"narrative": "...", "recommendation_type": "..."}}
"""

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiQuotaExceededError(RuntimeError):
    """Raised specifically for HTTP 429 -- distinct from a one-off malformed
    response (RuntimeError) so the caller (run_pipeline.py) can tell "this
    request happened to fail" apart from "every remaining request this run
    will fail too" and stop retrying per-cell instead of burning the full
    flagged-cell list one slow HTTP round-trip at a time (verified: a run
    with 1,404 flagged cells took ~25 minutes retrying a quota that was
    already exhausted on the first call)."""


def generate_narrative(grid_id: str, ews_code: int, vulnerability_index: float, matching_score: float) -> dict:
    payload = {
        "grid_id": grid_id, "ews_code": ews_code,
        "vulnerability_index": round(vulnerability_index, 3), "matching_score": round(matching_score, 1),
    }

    if not settings.gemini_api_key:
        return {
            "narrative": f"[STUB -- no GEMINI_API_KEY set] Grid {grid_id}: EWS {ews_code}, vulnerability {payload['vulnerability_index']}.",
            "recommendation_type": "pemantauan",
        }

    prompt = PROMPT_TEMPLATE.format(payload=json.dumps(payload))
    url = GEMINI_ENDPOINT.format(model=settings.gemini_model)

    try:
        resp = httpx.post(
            url, params={"key": settings.gemini_api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=90.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise GeminiQuotaExceededError(f"Gemini API returned 429: {e.response.text[:200]}")
        raise RuntimeError(f"Gemini API returned {e.response.status_code}: {e.response.text[:200]}")
    except httpx.RequestError as e:
        raise RuntimeError(f"Could not reach Gemini API: {e}")

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
        if "narrative" not in parsed or "recommendation_type" not in parsed:
            raise ValueError("Gemini response missing required fields")
        return parsed
    except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Malformed Gemini response, refusing to persist it: {e}")
