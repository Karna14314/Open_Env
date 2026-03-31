# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Code Review Environment — agent finds bugs in Python snippets.
3 tasks: syntax errors (easy) → logic bugs (medium) → security vulns (hard).
"""

from __future__ import annotations
import uuid
import random
from openenv.core.env_server.interfaces import Environment, Action, Observation
from models import ReviewAction, ReviewObservation, ReviewState
# ── Task bank ────────────────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "description": (
            "Review the following Python code and identify any syntax or "
            "runtime errors. Specify the bug type, the line number where "
            "the error occurs, and explain what is wrong."
        ),
        "snippets": [
            {
                "code": """\
def calculate_average(numbers)
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

result = calculate_average([10, 20, 30])
print(result)
""",
                "bug_type": "syntax",
                "line_number": 1,
                "keywords": ["colon", "missing", "def", "syntax"],
            },
            {
                "code": """\
def add_numbers(a, b)
    return a + b

result = add_numbers(5, 10)
print(result)
""",
                "bug_type": "syntax",
                "line_number": 1,
                "keywords": ["colon", "missing", "syntax"],
            },
            {
                "code": """\
def greet(name):
    message = "Hello " + name
    print(message)
    # Missing return statement here

greet("Alice")
""",
                "bug_type": "syntax",
                "line_number": 4,
                "keywords": ["return", "missing", "runtime"],
            },
            {
                "code": """\
def process_data(data):
    result = process(data)
    return result

result_value = process_data([1, 2, 3])
print(result_value)
""",
                "bug_type": "syntax",
                "line_number": 2,
                "keywords": ["undefined", "name", "not defined"],
            },
        ],
        "correct_bug_type": "syntax",
        "correct_keywords": ["colon", "missing", "def", "syntax"],
    },
    "medium": {
        "description": (
            "Review the following Python code. It runs without crashing "
            "but produces incorrect output. Identify the logic bug, "
            "the line number, and explain why it is wrong."
        ),
        "snippets": [
            {
                "code": """\
def find_max(numbers):
    max_val = numbers[0]
    for i in range(len(numbers)):
        if numbers[i] > max_val:
            max_val = numbers[i - 1]
    return max_val

print(find_max([3, 7, 2, 9, 4]))
""",
                "bug_type": "logic",
                "line_number": 5,
                "keywords": ["index", "i - 1", "off by one", "wrong", "logic"],
            },
            {
                "code": """\
def count_matches(lst, target):
    count = 0
    for item in lst:
        if item = target:
            count += 1
    return count

print(count_matches([1, 2, 2, 3, 2], 2))
""",
                "bug_type": "logic",
                "line_number": 4,
                "keywords": ["comparison", "==", "assignment", "operator"],
            },
            {
                "code": """\
def reverse_string(s):
    result = ""
    for i in range(len(s) - 1, -1, -1):
        for j in range(0, len(s)):
            result += s[j]
    return result

print(reverse_string("hello"))
""",
                "bug_type": "logic",
                "line_number": 3,
                "keywords": ["loop", "bounds", "reversed", "range"],
            },
            {
                "code": """\
def extract_substring(text):
    start = 5
    end = 2
    return text[start:end]

result = extract_substring("Hello World")
print(result)
""",
                "bug_type": "logic",
                "line_number": 4,
                "keywords": ["slice", "index", "string", "wrong"],
            },
        ],
        "correct_bug_type": "logic",
        "correct_keywords": ["index", "i - 1", "off by one", "wrong", "logic"],
    },
    "hard": {
        "description": (
            "Review the following Python code for security vulnerabilities. "
            "Identify the vulnerability type, the line number, and explain "
            "the security risk it introduces."
        ),
        "snippets": [
            {
                "code": """\
import sqlite3

def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()

user_input = input("Enter username: ")
print(get_user(user_input))
""",
                "bug_type": "security",
                "line_number": 6,
                "keywords": ["sql injection", "injection", "concatenat", "unsanitized", "parameterized"],
            },
            {
                "code": """\
def authenticate(username, password):
    admin_password = "SuperSecret123!"
    
    if username == "admin" and password == admin_password:
        return True
    return False

user_input = input("Username: ")
pass_input = input("Password: ")
print(authenticate(user_input, pass_input))
""",
                "bug_type": "security",
                "line_number": 2,
                "keywords": ["hardcoded", "password", "credentials", "secret", "plaintext"],
            },
            {
                "code": """\
import pickle

def deserialize_data(data):
    return pickle.loads(data)

received_data = input("Enter pickled data: ")
result = deserialize_data(received_data.encode())
print(result)
""",
                "bug_type": "security",
                "line_number": 4,
                "keywords": ["pickle", "untrusted", "deserialize", "arbitrary code"],
            },
            {
                "code": """\
def evaluate_expression(expr):
    result = eval(expr)
    return result

user_expr = input("Enter expression: ")
print(evaluate_expression(user_expr))
""",
                "bug_type": "security",
                "line_number": 2,
                "keywords": ["eval", "arbitrary code", "user input", "dangerous"],
            },
        ],
        "correct_bug_type": "security",
        "correct_keywords": ["sql injection", "injection", "concatenat", "unsanitized", "parameterized"],
    },
}

MAX_STEPS = 3

# ── Reward function ───────────────────────────────────────────────────────────

def compute_reward(action: ReviewAction, task: dict, attempt: int) -> tuple[float, str]:
    """
    Partial progress reward — not binary.
    Returns (reward_float, feedback_string).
    """
    reward = 0.0
    feedback_parts = []

    # Bug type match (+1.0)
    if action.bug_type.lower() == task["correct_bug_type"]:
        reward += 1.0
        feedback_parts.append("✓ Correct bug type identified.")
    else:
        reward -= 0.3
        feedback_parts.append(
            f"✗ Wrong bug type. Got '{action.bug_type}', "
            f"expected '{task['correct_bug_type']}'."
        )

    # Line number match (+0.5)
    if action.line_number == task["correct_line_number"]:
        reward += 0.5
        feedback_parts.append("✓ Correct line number.")
    else:
        feedback_parts.append(
            f"✗ Wrong line number. Got {action.line_number}, "
            f"expected {task['correct_line_number']}."
        )

    # Keyword quality check (+0.5)
    review_lower = action.review.lower()
    matched_keywords = [
        kw for kw in task["correct_keywords"] if kw in review_lower
    ]
    if matched_keywords:
        reward += 0.5
        feedback_parts.append(f"✓ Good explanation (matched: {matched_keywords}).")
    else:
        feedback_parts.append("✗ Explanation missing key concepts.")

    # Semantic similarity bonus (+0.25): if review length > 50 chars AND contains correct keyword
    if len(action.review) > 50 and matched_keywords:
        reward += 0.25
        feedback_parts.append("✓ Semantic similarity bonus: detailed and accurate explanation.")

    # Retry penalty
    if attempt > 1:
        penalty = 0.1 * (attempt - 1)
        reward -= penalty
        feedback_parts.append(f"⚠ Retry penalty: -{penalty:.1f}")

    # Clamp to 0.0-1.0 (max raw = 2.25, normalize)
    normalized = max(0.0, min(1.0, reward / 2.25))
    return round(normalized, 4), " ".join(feedback_parts)


# ── Environment ───────────────────────────────────────────────────────────────

class CodeReviewEnvironment(Environment):
    """
    Code Review Environment.
    Agent reviews Python snippets across 3 difficulty tasks.
    """

    def __init__(self):
        self._state = ReviewState()

    def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in TASKS:
            task_id = "easy"
        task = TASKS[task_id]
        # Randomly select a snippet from the available snippets
        selected_snippet = random.choice(task["snippets"])
        
        self._state = ReviewState(
            current_task_id=task_id,
            current_snippet=selected_snippet["code"],
            correct_bug_type=selected_snippet["bug_type"],
            correct_line_number=selected_snippet["line_number"],
            correct_keywords=selected_snippet["keywords"],
            step_count=0,
            task_episode_id=str(uuid.uuid4()),
            cumulative_reward=0.0,
            total_snippets=len(task["snippets"]),
        )
        return ReviewObservation(
            code_snippet=selected_snippet["code"],
            task_description=task["description"],
            task_id=task_id,
            attempt_number=0,
            previous_feedback="",
            done=False,
            hint=None,
        )

    def step(self, action: Action) -> Observation:
        if not isinstance(action, ReviewAction):
            raise ValueError(f"Expected ReviewAction, got {type(action)}")

        self._state.step_count += 1
        task_base = TASKS[self._state.current_task_id]
        
        # Create task dict with current snippet's correct answers for compute_reward
        task = {
            "correct_bug_type": self._state.correct_bug_type,
            "correct_line_number": self._state.correct_line_number,
            "correct_keywords": self._state.correct_keywords,
        }

        reward, feedback = compute_reward(
            action, task, self._state.step_count
        )
        self._state.cumulative_reward += reward

        done = (
            reward >= 0.75  # good enough answer
            or self._state.step_count >= MAX_STEPS
        )

        return ReviewObservation(
            code_snippet=self._state.current_snippet,
            task_description=task_base["description"],
            task_id=self._state.current_task_id,
            attempt_number=self._state.step_count,
            previous_feedback=feedback,
            done=done,
            hint=None,
        )

    @property
    def state(self) -> ReviewState:
        return self._state
