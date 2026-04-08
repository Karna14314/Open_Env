"""
envs/coding_env/models.py
--------------------------------
Action/Observation types for the Coding environment.
"""

from __future__ import annotations

from openenv.core.env_server.interfaces import Action, Observation, State


class CodeAction(Action):
    """Represents a single code-review submission."""

    review: str = ""
    file_path: str = ""
    issue_type: str = "logic"
    severity: str = "medium"
    bug_type: str = "none"
    line_number: int = -1
    confidence: float = 0.0
    # Optional fallback for compatibility with earlier code-exec flows.
    code: str = ""


class CodeObservation(Observation):
    """Observation returned by the code-review environment."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    task_id: str = ""
    difficulty: str = ""
    task_description: str = ""
    code_snippet: str = ""
    pr_title: str = ""
    pr_description: str = ""
    changed_files: str = ""
    previous_feedback: str = ""


class CodeState(State):
    """State for code-review episodes."""

    last_exit_code: int = 0
    task_id: str = ""
    difficulty: str = ""
    last_score: float = 0.0
