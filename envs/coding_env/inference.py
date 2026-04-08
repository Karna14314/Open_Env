#!/usr/bin/env python3
"""Hackathon baseline inference for coding_env.

MANDATORY environment variables handled here:
- API_BASE_URL (defaulted)
- MODEL_NAME (defaulted)
- HF_TOKEN (no default)
- LOCAL_IMAGE_NAME (optional, for local Docker workflows)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
BENCHMARK = os.getenv("BENCHMARK", "coding_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "1"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.60"))
MIN_STRICT_SCORE = 0.01
MAX_STRICT_SCORE = 0.99


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _strict_score(value: float) -> float:
    return max(MIN_STRICT_SCORE, min(MAX_STRICT_SCORE, round(float(value), 4)))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: str | None
) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={_bool_text(done)} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_bool_text(success)} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _safe_json(method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
    try:
        response = requests.request(method, url, timeout=30, **kwargs)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _task_list() -> List[str]:
    data = _safe_json("GET", f"{ENV_BASE_URL}/tasks")
    tasks = data.get("tasks", [])
    if isinstance(tasks, list):
        values: List[str] = []
        for item in tasks:
            if isinstance(item, dict) and item.get("task_id"):
                values.append(str(item["task_id"]))
        if values:
            return values
    return ["task_easy_1", "task_medium_1", "task_hard_1"]


def _build_action(client: OpenAI | None, task_description: str, code_snippet: str) -> Dict[str, Any]:
    fallback = {
        "review": "Likely logic issue in this PR change; please review line-level semantics.",
        "file_path": "services/metrics/aggregation.py",
        "issue_type": "logic",
        "severity": "medium",
        "bug_type": "logic",
        "line_number": 1,
        "confidence": 0.20,
    }

    if client is None:
        return fallback

    prompt = f"""You are reviewing a production pull request.
Task: {task_description}

PR context:
{code_snippet}

Return ONLY valid JSON with keys:
review (string),
file_path (string from changed files),
issue_type (one of logic|security|performance|maintainability),
severity (one of low|medium|high|critical),
bug_type (one of syntax|logic|security|none),
line_number (integer),
confidence (0.0-1.0 float)
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return fallback
        return {
            "review": str(parsed.get("review", fallback["review"])),
            "file_path": str(parsed.get("file_path", fallback["file_path"])),
            "issue_type": str(parsed.get("issue_type", fallback["issue_type"])),
            "severity": str(parsed.get("severity", fallback["severity"])),
            "bug_type": str(parsed.get("bug_type", fallback["bug_type"])),
            "line_number": int(parsed.get("line_number", fallback["line_number"])),
            "confidence": float(parsed.get("confidence", fallback["confidence"])),
        }
    except Exception:
        return fallback


def run_task(task_id: str, client: OpenAI | None) -> float:
    episode_id = f"baseline-{task_id}"
    rewards: List[float] = []
    score = MIN_STRICT_SCORE
    success = False
    last_error: str | None = None
    steps_taken = 0

    log_start(task_id, BENCHMARK, MODEL_NAME)

    try:
        reset_data = _safe_json(
            "POST",
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "episode_id": episode_id},
        )
        obs = reset_data.get("observation", {}) if isinstance(reset_data, dict) else {}
        task_description = str(obs.get("task_description", "Review code quality and bugs."))
        code_snippet = str(obs.get("code_snippet", ""))

        for step_num in range(1, MAX_STEPS + 1):
            action = _build_action(client, task_description, code_snippet)
            action_str = (
                f"file={action['file_path']};"
                f"issue={action['issue_type']};"
                f"sev={action['severity']};"
                f"bug_type={action['bug_type']};"
                f"line={action['line_number']};"
                f"confidence={float(action['confidence']):.2f}"
            )

            step_data = _safe_json(
                "POST",
                f"{ENV_BASE_URL}/step",
                json={
                    "action": action,
                    "task_id": task_id,
                    "episode_id": episode_id,
                },
            )
            reward = _strict_score(float(step_data.get("reward", MIN_STRICT_SCORE) or MIN_STRICT_SCORE))
            done = bool(step_data.get("done", not bool(step_data)))
            obs_after = step_data.get("observation", {}) if isinstance(step_data, dict) else {}
            raw_error = obs_after.get("last_action_error")
            last_error = str(raw_error) if raw_error else None

            rewards.append(reward)
            steps_taken = step_num
            log_step(step_num, action_str, reward, done, last_error)

            if done:
                break

        grader_data = _safe_json(
            "GET", f"{ENV_BASE_URL}/grader?task_id={task_id}&episode_id={episode_id}"
        )
        grader_score = _strict_score(float(grader_data.get("score", MIN_STRICT_SCORE) or MIN_STRICT_SCORE))
        step_score = _strict_score(rewards[-1] if rewards else MIN_STRICT_SCORE)
        score = _strict_score(max(grader_score, step_score))
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        last_error = str(exc)
        if steps_taken == 0:
            log_step(
                1,
                "bug_type=none;line=-1;confidence=0.00",
                MIN_STRICT_SCORE,
                True,
                last_error,
            )
            rewards.append(MIN_STRICT_SCORE)
            steps_taken = 1
        score = MIN_STRICT_SCORE
        success = False
    finally:
        log_end(success, max(1, steps_taken), score, rewards or [0.0])

    return score


def main() -> Dict[str, float]:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
    tasks = _task_list()

    scores: Dict[str, float] = {}
    for task_id in tasks:
        scores[task_id] = run_task(task_id, client)

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    scores["average"] = round(avg, 4)
    return scores


if __name__ == "__main__":
    main()
