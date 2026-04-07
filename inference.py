#!/usr/bin/env python3
"""Code Review Environment Baseline Evaluation.

Prints structured output format for programmatic consumption:
[START] task=<task_id>
[STEP] step=1 reward=<score>
[END] task=<task_id> score=<score> steps=1

This script evaluates baseline code review performance using an LLM.
"""

from __future__ import annotations

import json
import os
import requests
from typing import Dict

from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL = "gpt-4o-mini"

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# List of task IDs to evaluate
TASKS = os.getenv("TASKS", "task_1,task_2,task_3").split(",")

# ---------------------------------------------------------------------------
# Main Task Runner
# ---------------------------------------------------------------------------


def run_task(task_id: str) -> float:
    """Run a single code review task and return the score."""
    print(f"[START] task={task_id}", flush=True)
    
    # Reset environment
    reset_resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    obs = reset_resp.json()["observation"]
    
    code_snippet = obs["code_snippet"]
    task_description = obs["task_description"]
    
    # Call LLM
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
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        action = json.loads(raw)
    except Exception as e:
        print(f"LLM error for {task_id}: {e}", flush=True)
        action = {"review": "unknown", "bug_type": "none", "line_number": -1, "confidence": 0.0}

    # Step
    step_resp = requests.post(f"{BASE_URL}/step", json={"action": action})
    step_data = step_resp.json()
    feedback = step_data["observation"].get("previous_feedback", "")

    # Get grader score
    grader_resp = requests.get(f"{BASE_URL}/grader?task_id={task_id}&episode_id=baseline")
    score = grader_resp.json().get("score", 0.0)

    print(f"[STEP] step=1 reward={score}", flush=True)
    print(f"[END] task={task_id} score={score} steps=1", flush=True)
    
    return score


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    scores = {}
    for task_id in TASKS:
        scores[task_id] = run_task(task_id)
    
    average = round(sum(scores.values()) / len(scores), 4)
    scores["average"] = average
    
    print(f"\nBaseline Results: {json.dumps(scores, indent=2)}", flush=True)
    
    with open("baseline_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    
    return scores


if __name__ == "__main__":
    main()
