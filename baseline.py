#!/usr/bin/env python3
"""
Standalone baseline inference script.
Uses OpenAI gpt-4o-mini to review Python code across 3 difficulty levels.
Saves results to baseline_scores.json.
"""

import os
import json
import requests
from openai import OpenAI

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=api_key)

# Server endpoint
BASE_URL = "http://localhost:8000"
TASKS = ["easy", "medium", "hard"]

def reset_task(task_id: str) -> dict:
    """Reset environment for a given task_id."""
    response = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id}
    )
    response.raise_for_status()
    return response.json()

def step_task(action: dict) -> dict:
    """Submit action to environment and get observation."""
    response = requests.post(
        f"{BASE_URL}/step",
        json={"action": action}
    )
    response.raise_for_status()
    return response.json()

def review_code(code_snippet: str) -> dict:
    """Use GPT-4o-mini to review code snippet."""
    prompt = f"""Review this Python code. Reply as JSON with keys: review (str), bug_type (syntax/logic/security/none), line_number (int), confidence (float)

Code:
{code_snippet}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    content = response.choices[0].message.content
    
    # Try to extract JSON from response
    try:
        # First try direct JSON parsing
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON in the response text
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            result = json.loads(content[start:end])
        else:
            raise ValueError(f"Could not parse JSON from response: {content}")
    
    return result

def run_baseline():
    """Run baseline inference on all tasks."""
    results = {
        "scores": {},
        "details": {}
    }
    
    for task_id in TASKS:
        print(f"\n{'='*60}")
        print(f"Running task: {task_id}")
        print('='*60)
        
        # Reset environment
        obs = reset_task(task_id)
        code_snippet = obs.get("code_snippet", "")
        print(f"Code snippet:\n{code_snippet}\n")
        
        # Get review from GPT-4o-mini
        print("Calling GPT-4o-mini for review...")
        review_result = review_code(code_snippet)
        print(f"Review result: {review_result}")
        
        # Prepare action
        action = {
            "review": review_result.get("review", ""),
            "bug_type": review_result.get("bug_type", "none"),
            "line_number": int(review_result.get("line_number", -1)),
            "confidence": float(review_result.get("confidence", 0.0))
        }
        
        # Submit action to environment
        print(f"Submitting action: {action}")
        step_obs = step_task(action)
        
        # Extract score from observation
        # The step response should have reward/score information
        score = step_obs.get("cumulative_reward", 0.0)
        feedback = step_obs.get("previous_feedback", "")
        
        print(f"Score: {score}")
        print(f"Feedback: {feedback}")
        
        results["scores"][task_id] = score
        results["details"][task_id] = {
            "action": action,
            "feedback": feedback,
            "score": score
        }
    
    # Calculate average
    scores = list(results["scores"].values())
    average = sum(scores) / len(scores) if scores else 0.0
    results["average"] = round(average, 4)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print('='*60)
    for task_id in TASKS:
        print(f"{task_id:10s}: {results['scores'][task_id]:.4f}")
    print(f"{'Average':10s}: {results['average']:.4f}")
    print('='*60 + "\n")
    
    # Save to file
    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to baseline_scores.json")
    return results

if __name__ == "__main__":
    run_baseline()
