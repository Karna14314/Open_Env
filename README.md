---
title: Code Review Environment
emoji: 🎯
colorFrom: pink
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Code Review Environment

An OpenEnv environment where an AI agent reviews Python code snippets to identify bugs across three difficulty levels.

🤗 **Space:** https://huggingface.co/spaces/ncncomplete/code-review-env

## Environment Description

The agent receives a Python code snippet and must identify the bug type, line number, and provide an explanation. The environment simulates real-world code review tasks that developers perform daily.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| easy | Easy | Identify syntax/runtime errors |
| medium | Medium | Identify logic bugs in code that runs but produces wrong output |
| hard | Hard | Identify security vulnerabilities |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| review | str | Written analysis of the code |
| bug_type | str | One of: syntax, logic, security, none |
| line_number | int | Line number where bug occurs (-1 if unknown) |
| confidence | float | Agent confidence 0.0–1.0 |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| code_snippet | str | Python code to review |
| task_description | str | What the agent is asked to do |
| task_id | str | easy, medium, or hard |
| attempt_number | int | Steps taken so far |
| previous_feedback | str | Feedback from last step |
| done | bool | Whether episode is complete |

## Reward Function

- **+1.0** correct bug type identified
- **+0.5** correct line number identified
- **+0.5** quality explanation (key concepts present)
- **-0.3** wrong bug category confidently stated
- **-0.1** per retry after first attempt
- Normalized to 0.0–1.0 range

## Baseline Scores

| Task | Score |
|------|-------|
| easy | 1.0 |
| medium | 1.0 |
| hard | 1.0 |
| **average** | **1.0** |

## Setup

```bash
pip install openenv-core fastapi uvicorn pydantic openai
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /reset` — Start new episode with `{"task_id": "easy|medium|hard"}`
- `POST /step` — Submit action with `{"action": {...}}`
- `GET /state` — Get current environment state
- `GET /tasks` — List all tasks and action schema
- `GET /grader` — Get grader score for a task
- `GET /baseline` — Run baseline inference on all tasks
