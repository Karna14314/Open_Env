# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Code review environment with task-based grading and normalized rewards."""

import uuid
from typing import Any

from openenv.core.env_server.interfaces import Action, Environment, Observation

from ..models import CodeAction, CodeObservation, CodeState
from .task_bank import (
    format_task_prompt,
    get_task,
    grade_action,
    list_tasks,
    record_episode_score,
)


class PythonCodeActEnv(Environment):
    """
    Task-driven code-review environment.

    Episodes are single-step:
    1. `reset(task_id=...)` returns a code snippet + task description.
    2. Agent submits CodeAction(review, bug_type, line_number, confidence).
    3. `step()` returns graded reward in [0.0, 1.0] and done=True.
    """

    def __init__(
        self,
    ):
        super().__init__(transform=None)
        self._state = CodeState()
        self._current_task_id = "task_easy_1"

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset environment and pick a task (easy/medium/hard).
        """
        requested_task_id = kwargs.get("task_id", self._current_task_id)
        task = get_task(str(requested_task_id))
        self._current_task_id = task.task_id

        self._state = CodeState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task.task_id,
            difficulty=task.difficulty,
            last_score=0.0,
        )
        self._state.last_exit_code = 0

        observation = CodeObservation(
            stdout="Task initialized.",
            stderr="",
            exit_code=0,
            task_id=task.task_id,
            difficulty=task.difficulty,
            task_description=(
                "Review this pull request and report the highest-impact issue "
                "with file_path, issue_type, severity, line_number, and evidence."
            ),
            code_snippet=format_task_prompt(task),
            pr_title=task.pr_title,
            pr_description=task.pr_description,
            changed_files="\n".join(task.changed_files),
            previous_feedback="",
            done=False,
            reward=0.0,
            metadata={"available_tasks": list_tasks()},
        )

        return self._apply_transform(observation)

    def step(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute code action and return observation.

        Args:
            action: CodeAction containing the code to execute

        Returns:
            CodeObservation with execution results (stdout, stderr, exit_code)

        Raises:
            ValueError: If action is not a CodeAction instance
        """
        if not isinstance(action, CodeAction):
            raise ValueError(f"Expected CodeAction, got {type(action)}")

        requested_task_id = kwargs.get("task_id")
        task_id = str(requested_task_id or self._state.task_id or self._current_task_id)
        task = get_task(task_id)
        episode_id = str(
            kwargs.get("episode_id") or self._state.episode_id or str(uuid.uuid4())
        )

        self._state.task_id = task.task_id
        self._state.difficulty = task.difficulty
        self._state.episode_id = episode_id
        reward, feedback = grade_action(action, task)

        self._state.step_count += 1
        self._state.last_exit_code = 0
        self._state.last_score = reward
        record_episode_score(task.task_id, episode_id, reward)

        observation = CodeObservation(
            stdout=feedback,
            stderr="",
            exit_code=0,
            task_id=task.task_id,
            difficulty=task.difficulty,
            task_description=(
                "Review this pull request and report the highest-impact issue "
                "with file_path, issue_type, severity, line_number, and evidence."
            ),
            code_snippet=format_task_prompt(task),
            pr_title=task.pr_title,
            pr_description=task.pr_description,
            changed_files="\n".join(task.changed_files),
            previous_feedback=feedback,
            reward=reward,
            done=True,
        )

        return self._apply_transform(observation)

    @property
    def state(self) -> CodeState:
        """Get current environment state including last exit code."""
        return self._state
