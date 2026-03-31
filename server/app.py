# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI server for the Code Review Environment.
"""

from models import ReviewAction, ReviewObservation
from server.code_review_env_environment import CodeReviewEnvironment
from openenv.core.env_server import create_app
from fastapi import FastAPI, Query
from fastapi.routing import APIRouter

app = create_app(
    CodeReviewEnvironment,
    ReviewAction,
    ReviewObservation,
    env_name="code_review_env",
)

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": "easy",
                "description": "Identify syntax/runtime errors in Python code",
                "difficulty": "easy",
                "action_schema": {
                    "review": "string - your analysis",
                    "bug_type": "string - syntax | logic | security | none",
                    "line_number": "int - line with the bug, -1 if unknown",
                    "confidence": "float - your confidence 0.0 to 1.0"
                }
            },
            {
                "task_id": "medium",
                "description": "Identify logic bugs in code that runs but produces wrong output",
                "difficulty": "medium",
                "action_schema": {
                    "review": "string - your analysis",
                    "bug_type": "string - syntax | logic | security | none",
                    "line_number": "int - line with the bug, -1 if unknown",
                    "confidence": "float - your confidence 0.0 to 1.0"
                }
            },
            {
                "task_id": "hard",
                "description": "Identify security vulnerabilities in Python code",
                "difficulty": "hard",
                "action_schema": {
                    "review": "string - your analysis",
                    "bug_type": "string - syntax | logic | security | none",
                    "line_number": "int - line with the bug, -1 if unknown",
                    "confidence": "float - your confidence 0.0 to 1.0"
                }
            }
        ]
    }

@app.get("/grader")
def grader(task_id: str = Query("easy"), episode_id: str = Query(None)):
    """
    Run a single task with a perfect answer.
    Query params: task_id (str), episode_id (str, optional)
    Returns: {"task_id": str, "score": float, "feedback": str}
    """
    env = CodeReviewEnvironment()
    env.reset(task_id)
    
    # Create perfect answer based on task_id
    if task_id == "easy":
        action = ReviewAction(
            review="Line 1 is missing a colon after the function definition. This is a syntax error.",
            bug_type="syntax",
            line_number=1,
            confidence=0.95
        )
    elif task_id == "medium":
        action = ReviewAction(
            review="Line 5 has an index error: it should be max_val = numbers[i], not numbers[i - 1]. This is a logic bug.",
            bug_type="logic",
            line_number=5,
            confidence=0.95
        )
    else:  # hard
        action = ReviewAction(
            review="Line 6 has a SQL injection vulnerability because the username is concatenated directly into the query without parameterized statements.",
            bug_type="security",
            line_number=6,
            confidence=0.95
        )
    
    obs = env.step(action)
    return {
        "task_id": task_id,
        "score": env.state.cumulative_reward,
        "feedback": obs.previous_feedback
    }

@app.get("/baseline")
def baseline():
    """
    Run all 3 tasks (easy, medium, hard) with perfect hardcoded answers.
    Returns: {"scores": {"easy": float, "medium": float, "hard": float}, "average": float}
    """
    scores = {}
    
    for task_id in ["easy", "medium", "hard"]:
        env = CodeReviewEnvironment()
        env.reset(task_id)
        
        # Create perfect answer based on task_id
        if task_id == "easy":
            action = ReviewAction(
                review="Line 1 is missing a colon after the function definition. This is a syntax error.",
                bug_type="syntax",
                line_number=1,
                confidence=0.95
            )
        elif task_id == "medium":
            action = ReviewAction(
                review="Line 5 has an index error: it should be max_val = numbers[i], not numbers[i - 1]. This is a logic bug.",
                bug_type="logic",
                line_number=5,
                confidence=0.95
            )
        else:  # hard
            action = ReviewAction(
                review="Line 6 has a SQL injection vulnerability because the username is concatenated directly into the query without parameterized statements.",
                bug_type="security",
                line_number=6,
                confidence=0.95
            )
        
        obs = env.step(action)
        scores[task_id] = env.state.cumulative_reward
    
    average = sum(scores.values()) / len(scores)
    return {
        "scores": scores,
        "average": round(average, 4)
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
