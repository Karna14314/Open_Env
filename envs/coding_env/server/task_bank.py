"""PR-triage task definitions and grading utilities for coding_env."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Dict, List, Tuple

try:
    from coding_env.models import CodeAction
except ImportError:
    from ..models import CodeAction


@dataclass(frozen=True)
class CodeReviewTask:
    task_id: str
    difficulty: str
    pr_title: str
    pr_description: str
    changed_files: Tuple[str, ...]
    unified_diff: str
    expected_file_path: str
    expected_issue_type: str
    expected_bug_type: str
    expected_severity: str
    expected_line_number: int
    expected_keywords: Tuple[str, ...]


TASKS: Dict[str, CodeReviewTask] = {
    "task_easy_1": CodeReviewTask(
        task_id="task_easy_1",
        difficulty="easy",
        pr_title="Fix average response time aggregation in metrics service",
        pr_description="Refactor aggregation logic and simplify average calculation.",
        changed_files=("services/metrics/aggregation.py",),
        unified_diff=(
            "diff --git a/services/metrics/aggregation.py b/services/metrics/aggregation.py\n"
            "@@ -10,6 +10,6 @@ def compute_avg(latencies):\n"
            "     total = 0\n"
            "     for latency in latencies:\n"
            "         total += latency\n"
            "-    return total / len(total)\n"
            "+    return total / len(total)\n"
        ),
        expected_file_path="services/metrics/aggregation.py",
        expected_issue_type="logic",
        expected_bug_type="logic",
        expected_severity="medium",
        expected_line_number=4,
        expected_keywords=("len(total)", "len(latencies)", "typeerror"),
    ),
    "task_medium_1": CodeReviewTask(
        task_id="task_medium_1",
        difficulty="medium",
        pr_title="Optimize login query path in auth service",
        pr_description="Use direct SQL construction for faster username/password checks.",
        changed_files=("services/auth/login.py",),
        unified_diff=(
            "diff --git a/services/auth/login.py b/services/auth/login.py\n"
            "@@ -21,7 +21,7 @@ def login(conn, username, password):\n"
            "-    query = \"SELECT * FROM users WHERE name=? AND pw=?\"\n"
            "-    return conn.execute(query, (username, password)).fetchone() is not None\n"
            "+    query = f\"SELECT * FROM users WHERE name='{username}' AND pw='{password}'\"\n"
            "+    return conn.execute(query).fetchone() is not None\n"
        ),
        expected_file_path="services/auth/login.py",
        expected_issue_type="security",
        expected_bug_type="security",
        expected_severity="high",
        expected_line_number=2,
        expected_keywords=("sql injection", "parameterized", "prepared statement"),
    ),
    "task_hard_1": CodeReviewTask(
        task_id="task_hard_1",
        difficulty="hard",
        pr_title="Add cache layer to user profile fetch endpoint",
        pr_description="Protect cache updates with lock to avoid races and keep data coherent.",
        changed_files=("services/profile/cache_layer.py",),
        unified_diff=(
            "diff --git a/services/profile/cache_layer.py b/services/profile/cache_layer.py\n"
            "@@ -4,12 +4,12 @@ lock = Lock()\n"
            " cache = {}\n"
            "\n"
            " def get_user(user_id, db):\n"
            "     with lock:\n"
            "         if user_id in cache:\n"
            "             return cache[user_id]\n"
            "         data = db.fetch_user(user_id)\n"
            "         cache[user_id] = data\n"
            "         return data\n"
        ),
        expected_file_path="services/profile/cache_layer.py",
        expected_issue_type="performance",
        expected_bug_type="logic",
        expected_severity="high",
        expected_line_number=7,
        expected_keywords=("lock contention", "critical section", "latency"),
    ),
}


EPISODE_SCORES: Dict[tuple[str, str], float] = {}
SCORE_FILE = os.getenv("CODING_ENV_SCORE_FILE", "/tmp/coding_env_episode_scores.json")
MIN_STRICT_SCORE = 0.01
MAX_STRICT_SCORE = 0.99


def list_tasks() -> List[Dict[str, str]]:
    """Return public task metadata."""
    return [
        {"task_id": t.task_id, "difficulty": t.difficulty, "pr_title": t.pr_title}
        for t in sorted(TASKS.values(), key=lambda item: item.task_id)
    ]


def get_task(task_id: str) -> CodeReviewTask:
    """Resolve task by id."""
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available tasks: {', '.join(sorted(TASKS))}"
        )
    return TASKS[task_id]


def format_task_prompt(task: CodeReviewTask) -> str:
    """Format a realistic PR-review prompt."""
    files = "\n".join(f"- {path}" for path in task.changed_files)
    return (
        f"PR Title: {task.pr_title}\n"
        f"PR Description: {task.pr_description}\n"
        f"Changed Files:\n{files}\n\n"
        f"Unified Diff:\n{task.unified_diff}\n\n"
        "Review objective: identify the highest-impact issue and provide "
        "file path, issue type, severity, and exact line."
    )


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _action_issue_type(action: CodeAction) -> str:
    issue_type = getattr(action, "issue_type", "")
    if issue_type:
        return str(issue_type)
    return str(action.bug_type)


def grade_action(action: CodeAction, task: CodeReviewTask) -> tuple[float, str]:
    """Score PR-triage action in strict (0, 1) with partial credit."""
    score = 0.0
    parts: List[str] = []

    file_path = str(getattr(action, "file_path", "") or "")
    if _normalize(file_path) == _normalize(task.expected_file_path):
        score += 0.30
        parts.append("file_path matched (+0.30)")
    else:
        parts.append(
            f"file_path mismatch (expected {task.expected_file_path}, got {file_path or 'none'})"
        )

    issue_type = _action_issue_type(action)
    if _normalize(issue_type) == _normalize(task.expected_issue_type):
        score += 0.30
        parts.append("issue_type matched (+0.30)")
    elif _normalize(action.bug_type) == _normalize(task.expected_bug_type):
        score += 0.20
        parts.append("bug_type matched (+0.20)")
    else:
        parts.append(
            f"issue mismatch (expected {task.expected_issue_type}/{task.expected_bug_type}, got {issue_type}/{action.bug_type})"
        )

    severity = str(getattr(action, "severity", "") or "")
    if _normalize(severity) == _normalize(task.expected_severity):
        score += 0.20
        parts.append("severity matched (+0.20)")
    elif severity:
        score += 0.10
        parts.append("severity provided but not exact (+0.10)")
    else:
        parts.append("severity missing (+0.00)")

    if action.line_number == task.expected_line_number:
        score += 0.10
        parts.append("line_number matched (+0.10)")
    elif abs(action.line_number - task.expected_line_number) <= 2:
        score += 0.05
        parts.append("line_number near miss (+0.05)")
    else:
        parts.append(
            f"line_number mismatch (expected {task.expected_line_number}, got {action.line_number})"
        )

    review_text = (action.review or "").lower()
    keyword_hits = sum(1 for kw in task.expected_keywords if kw.lower() in review_text)
    if keyword_hits > 0:
        keyword_bonus = min(0.09, keyword_hits * 0.03)
        score += keyword_bonus
        parts.append(f"evidence quality matched (+{keyword_bonus:.2f})")
    else:
        parts.append("insufficient evidence language (+0.00)")

    score = _to_strict_open_score(score)
    return score, "; ".join(parts)


def record_episode_score(task_id: str, episode_id: str, score: float) -> None:
    """Persist normalized score for grader endpoint."""
    normalized = _to_strict_open_score(score)
    EPISODE_SCORES[(task_id, episode_id)] = normalized
    _persist_score(task_id, episode_id, normalized)


def get_episode_score(task_id: str, episode_id: str) -> float:
    """Read score for task/episode pair."""
    in_memory = EPISODE_SCORES.get((task_id, episode_id))
    if in_memory is not None:
        return in_memory
    return _load_persisted_score(task_id, episode_id)


def _persist_score(task_id: str, episode_id: str, score: float) -> None:
    key = f"{task_id}::{episode_id}"
    payload: Dict[str, float] = {}
    if os.path.exists(SCORE_FILE):
        try:
            with open(SCORE_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                payload = {
                    str(k): float(v)
                    for k, v in loaded.items()
                    if isinstance(v, (int, float))
                }
        except Exception:
            payload = {}

    payload[key] = float(score)
    with open(SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _load_persisted_score(task_id: str, episode_id: str) -> float:
    if not os.path.exists(SCORE_FILE):
        return MIN_STRICT_SCORE
    try:
        with open(SCORE_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        key = f"{task_id}::{episode_id}"
        value = loaded.get(key, 0.0) if isinstance(loaded, dict) else 0.0
        return _to_strict_open_score(value)
    except Exception:
        return MIN_STRICT_SCORE


def _to_strict_open_score(value: float) -> float:
    """Clamp to strict open interval (0, 1)."""
    return max(MIN_STRICT_SCORE, min(MAX_STRICT_SCORE, round(float(value), 4)))
