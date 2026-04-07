#!/usr/bin/env python3
"""Solve a coding task with a hosted LLM via Hugging Face Inference.

Prints structured output format for programmatic consumption:
[START] task=coding
[STEP] step=1 reward=0.5
[END] task=coding score=1.0 steps=1

This script is based on examples/coding_env_inference.py with structured output.

Prerequisites
-------------
1. Build the Coding environment Docker image::

       docker build \
           -f envs/coding_env/server/Dockerfile \
           -t coding-env:latest .

2. Set your Hugging Face token, or any other API key that is compatible with the OpenAI API:

       export HF_TOKEN=your_token_here
       export API_KEY=your_api_key_here

3. Run the script::

       python inference.py
"""

from __future__ import annotations

import os
import re
from typing import Tuple

from openai import OpenAI

from coding_env import CodeAction, CodingEnv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "https://router.huggingface.co/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

MODEL = "openai/gpt-oss-120b:novita"
MAX_STEPS = 5
VERBOSE = False

CODING_TASK = (
    "Write Python code that prints the sum of squares of the integers from 1 "
    "to 100 inclusive. The final line must be exactly `Result: <value>` with "
    "the correct number substituted."
)
EXPECTED_SUBSTRING = "Result: 338350"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Respond with valid Python code that "
    "solves the user's task. Always wrap your final answer in a fenced code "
    "block starting with ```python. Provide a complete script that can be "
    "executed as-is, with no commentary outside the code block."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_python_code(text: str) -> str:
    """Extract the first Python code block from the model output."""

    code_blocks = re.findall(
        r"```(?:python)?\s*(.*?)```",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if code_blocks:
        return code_blocks[0].strip()
    return text.strip()


def format_feedback(
    step: int,
    stdout: str,
    stderr: str,
    exit_code: int,
) -> str:
    """Generate feedback text describing the previous execution."""

    stdout_display = stdout if stdout.strip() else "<empty>"
    stderr_display = stderr if stderr.strip() else "<empty>"
    return (
        f"Execution feedback for step {step}:\n"
        f"exit_code={exit_code}\n"
        f"stdout:\n{stdout_display}\n"
        f"stderr:\n{stderr_display}\n"
        "If the task is not solved, return an improved Python script."
    )


def build_initial_prompt(task: str) -> str:
    """Construct the first user prompt for the coding task."""

    return (
        "You must write Python code to satisfy the following task. "
        "When executed, your script should behave exactly as described.\n\n"
        f"Task:\n{task}\n\n"
        "Reply with the full script in a single ```python code block."
    )


# ---------------------------------------------------------------------------
# Gameplay
# ---------------------------------------------------------------------------


def solve_coding_task(
    env: CodingEnv,
    client: OpenAI,
) -> Tuple[bool, int]:
    """Iteratively ask the model for code until the task is solved.
    
    Returns:
        Tuple of (solved, steps_taken)
    """

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_prompt(CODING_TASK)},
    ]

    obs = env.reset().observation

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=2048,
            temperature=0.2,
        )

        assistant_message = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": assistant_message})

        code = extract_python_code(assistant_message)

        result = env.step(CodeAction(code=code))
        obs = result.observation

        # Compute reward: 1.0 if solved, 0.5 if code executed successfully, 0.0 if error
        if obs.exit_code == 0 and EXPECTED_SUBSTRING in obs.stdout:
            reward = 1.0
        elif obs.exit_code == 0:
            reward = 0.5
        else:
            reward = 0.0

        print(f"[STEP] step={step} reward={reward}", flush=True)

        # Check if solved
        solved = obs.exit_code == 0 and EXPECTED_SUBSTRING in obs.stdout
        if solved:
            return True, step

        history.append(
            {
                "role": "user",
                "content": format_feedback(
                    step,
                    obs.stdout,
                    obs.stderr,
                    obs.exit_code,
                ),
            }
        )

        # Keep conversation history compact to avoid exceeding context limits
        if len(history) > 20:
            history = [history[0]] + history[-19:]

    return False, MAX_STEPS


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "HF_TOKEN (or API_KEY) must be set to query the model."
        )

    print("[START] task=coding", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = CodingEnv.from_docker_image(
        "coding-env:latest",
        ports={8000: 8000},
    )

    try:
        success, steps_taken = solve_coding_task(env, client)
    finally:
        env.close()

    # Print final result
    score = 1.0 if success else 0.0
    print(f"[END] task=coding score={score} steps={steps_taken}", flush=True)


if __name__ == "__main__":
    main()
