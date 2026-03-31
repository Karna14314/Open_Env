# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Code Review Environment.
Agent receives Python code snippets and must identify bugs.
"""

from __future__ import annotations
from typing import Optional
from openenv.core.env_server.interfaces import Action, Observation, State


class ReviewAction(Action):
    """Action taken by the agent to review a code snippet."""
    review: str                    # agent's written analysis
    bug_type: str                  # "syntax" | "logic" | "security" | "none"
    line_number: int               # which line has the issue, -1 if unknown
    confidence: float              # agent's confidence 0.0-1.0


class ReviewObservation(Observation):
    """What the agent sees at each step."""
    code_snippet: str              # the Python code to review
    task_description: str          # what the agent is asked to do
    task_id: str                   # "easy" | "medium" | "hard"
    attempt_number: int            # how many steps taken so far
    previous_feedback: str         # feedback from last step, empty on reset
    done: bool                     # whether episode is complete
    hint: Optional[str] = None     # optional hint for the agent


class ReviewState(State):
    """Internal environment state."""
    current_task_id: str = "easy"
    current_snippet: str = ""
    correct_bug_type: str = ""
    correct_line_number: int = -1
    correct_keywords: list = []
    step_count: int = 0
    task_episode_id: str = ""
    cumulative_reward: float = 0.0
    total_snippets: int = 4
