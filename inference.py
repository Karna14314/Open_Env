"""
Baseline inference script for code_review_env.
Uses OpenAI API to run an agent against all 3 tasks.
"""
import os
import json
import requests
from openai import OpenAI

BASE_URL = os.getenv("ENV_URL", "http://localhost:8000")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key"))

TASKS = ["easy", "medium", "hard"]

def run_task(task_id: str) -> float:
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
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        action = json.loads(raw)
    except Exception as e:
        print(f"LLM error for {task_id}: {e}")
        action = {"review": "unknown", "bug_type": "none", "line_number": -1, "confidence": 0.0}

    # Step
    step_resp = requests.post(f"{BASE_URL}/step", json={"action": action})
    step_data = step_resp.json()
    
    # Get grader score
    grader_resp = requests.get(f"{BASE_URL}/grader?task_id={task_id}&episode_id=baseline")
    score = grader_resp.json().get("score", 0.0)
    
    print(f"Task: {task_id} | Score: {score} | Feedback: {step_data['observation'].get('previous_feedback', '')}")
    return score

def main():
    scores = {}
    for task_id in TASKS:
        scores[task_id] = run_task(task_id)
    
    average = sum(scores.values()) / len(scores)
    scores["average"] = round(average, 4)
    
    print(f"\nBaseline Results: {json.dumps(scores, indent=2)}")
    
    with open("baseline_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    
    return scores

if __name__ == "__main__":
    main()
