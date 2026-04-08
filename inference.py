#!/usr/bin/env python3
"""Code Review Environment Baseline Evaluation.

This script is hardened for validator compatibility:
- Always prints [START]/[STEP]/[END] to stdout with flush=True
- Avoids failing before first [START] due to optional deps/credentials
- Never redirects stdout
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    import requests
except Exception:
    requests = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL = "gpt-4o-mini"

# List of task IDs to evaluate
TASKS = os.getenv("TASKS", "task_1,task_2,task_3").split(",")

# ---------------------------------------------------------------------------
# Main Task Runner
# ---------------------------------------------------------------------------


def _build_action(task_description: str, code_snippet: str) -> Dict[str, Any]:
    """Build an action via LLM when available; otherwise return safe fallback."""
    fallback_action: Dict[str, Any] = {
        "review": "Unable to run model; submitting safe fallback review.",
        "bug_type": "none",
        "line_number": -1,
        "confidence": 0.0,
    }

    if not API_KEY:
        return fallback_action

    try:
        from openai import OpenAI  # Lazy import to avoid failing at module import time

        client = OpenAI(api_key=API_KEY)
    except Exception:
        return fallback_action

    prompt = f"""You are a code reviewer. {task_description}

Code to review:
```python
{code_snippet}
```

Respond ONLY with valid JSON, no markdown:
{{
  "review": "your detailed analysis",
  "bug_type": "syntax or logic or security or none",
  "line_number": <integer>,
  "confidence": <float 0.0-1.0>
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return fallback_action
    except Exception:
        return fallback_action


def _safe_post_json(url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return JSON body or None on any network/JSON failure."""
    if requests is None:
        return None
    try:
        response = requests.post(url, json=payload, timeout=30)
        return response.json()
    except Exception:
        return None


def _safe_get_json(url: str) -> Optional[Dict[str, Any]]:
    """Return JSON body or None on any network/JSON failure."""
    if requests is None:
        return None
    try:
        response = requests.get(url, timeout=30)
        return response.json()
    except Exception:
        return None


def run_task(task_id: str) -> float:
    """Run a single code review task and return the score."""
    print(f"[START] task={task_id}", flush=True)

    score = 0.0
    steps = 1

    reset_data = _safe_post_json(f"{BASE_URL}/reset", {"task_id": task_id}) or {}
    obs = reset_data.get("observation", {}) if isinstance(reset_data, dict) else {}

    code_snippet = obs.get("code_snippet", "")
    task_description = obs.get("task_description", "Review the provided code.")
    action = _build_action(str(task_description), str(code_snippet))

    # If stepping fails, we still emit structured output with reward=0.0
    _safe_post_json(f"{BASE_URL}/step", {"action": action})

    grader_data = _safe_get_json(f"{BASE_URL}/grader?task_id={task_id}&episode_id=baseline") or {}
    if isinstance(grader_data, dict):
        try:
            score = float(grader_data.get("score", 0.0))
        except Exception:
            score = 0.0

    print(f"[STEP] step=1 reward={score}", flush=True)
    print(f"[END] task={task_id} score={score} steps={steps}", flush=True)

    return score


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    scores = {}
    normalized_tasks = [t.strip() for t in TASKS if t.strip()]
    if not normalized_tasks:
        normalized_tasks = ["task_1"]

    for task_id in normalized_tasks:
        scores[task_id] = run_task(task_id)

    average = round(sum(scores.values()) / len(scores), 4)
    scores["average"] = average

    print(f"\nBaseline Results: {json.dumps(scores, indent=2)}", flush=True)

    with open("baseline_scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    return scores


if __name__ == "__main__":
    main()
