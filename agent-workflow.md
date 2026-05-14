# Agent Workflow: Ad Click Prediction MLOps

This repository enforces strict **Anti-Slop** engineering standards. All human developers and AI coding agents must adhere to the following workflow and quality gates.

## 1. RPI Protocol (Research - Plan - Implement)
Do not attempt to research, design, and code complex features in a single context window.
- **Research Phase**: Read files, understand context, do NOT write code.
- **Plan Phase**: Write a step-by-step implementation plan and review it.
- **Implement Phase**: Execute the plan in a fresh context window.

## 2. Quality Gates
Code cannot be committed unless it passes all gates.
- **Testing**: `pytest tests/` must pass 100%. No skipped tests without a tracked issue.
- **Linting**: `ruff check .` must return zero warnings. We use a strict ruleset (E, F, I, N, UP, B, A, SIM, TCH).
- **Type Checking**: `mypy .` must pass. All function signatures require strict typing.
- **Formatting**: `ruff format` dictates all code style.
- **Pre-commit**: All gates run automatically via `.pre-commit-config.yaml`.

## 3. Agent Isolation
- **Branching Strategy**: Agents work on isolated feature branches (e.g., `feature/add-drift-detection`).
- **No Force Pushes**: `git push --force` is strictly prohibited.
- **Traceability**: If an agent wrote the code, the commit message or PR description should note it.

## 4. Anti-Slop Checklist
- [ ] Are all tests passing (100%, no skips)?
- [ ] Is strict linting enabled (zero warnings)?
- [ ] Is type checking passing?
- [ ] Did you avoid mocking the database/dataframes? (Integration tests > Mocks).
- [ ] Are the APIs using Pydantic BaseModels instead of raw Dictionaries?
- [ ] Was the implementation plan reviewed by a human *before* writing the code?
- [ ] Did you run `pre-commit run --all-files` before finalizing the task?