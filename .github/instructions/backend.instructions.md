---
applyTo: "backend/**/*.py"
description: "Use when changing FastAPI backend code, exercise modules, state machine, classifier, smoothing, rep counting, or upload routes."
---

# Backend Instructions

## Focus
- Treat backend as the source of truth for this repository.
- Treat local frontend code as a placeholder only.
- Preserve API and WebSocket response compatibility for the external Next.js PWA client.

## Core Boundaries
- API/WebSocket endpoints: [backend/api/routes.py](../../backend/api/routes.py), [backend/api/upload.py](../../backend/api/upload.py)
- Session state transitions: [backend/state_machine/manager.py](../../backend/state_machine/manager.py)
- Exercise classification: [backend/exercises/classifier.py](../../backend/exercises/classifier.py)
- Exercise plugin contract and shared math: [backend/exercises/base.py](../../backend/exercises/base.py)
- Counting and smoothing utilities: [backend/utils/rep_counter.py](../../backend/utils/rep_counter.py), [backend/utils/smoothing.py](../../backend/utils/smoothing.py)

## Change Rules
- Keep changes minimal and module-scoped; avoid broad refactors.
- For new exercise logic, add or update files under [backend/exercises/](../../backend/exercises/).
- Keep thresholds and heuristics in the exercise module where they are used.
- Return actionable violations and corrections in every analysis path.
- Do not silently rename or remove response fields used by clients.

## Realtime Contract Safety
- Preserve the landmarks-in, feedback-out WebSocket flow in [backend/api/routes.py](../../backend/api/routes.py).
- If response fields must change, update [README.md](../../README.md) and document migration impact.
- Keep rep-phase and rep-count behavior compatible with hysteresis logic.

## Validation
- Install deps: pip install -r backend/requirements.txt
- Run API: cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
- Validate root and health endpoints.
- Validate at least one WebSocket roundtrip with realistic landmarks.
- If upload flow changes, validate init -> chunk(s) -> complete in [backend/api/upload.py](../../backend/api/upload.py).
