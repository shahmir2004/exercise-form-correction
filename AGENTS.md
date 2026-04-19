# AGENTS.md

Guidance for AI coding agents working in this repository.

## Scope
- Keep changes focused and minimal.
- Prefer updating existing modules over broad refactors.
- If behavior changes, update docs in [README.md](README.md).

## Primary Docs
- Project overview, setup, API contract, and roadmap: [README.md](README.md)
- Backend config values and defaults: [backend/config/settings.py](backend/config/settings.py)
- Frontend runtime config: [frontend/src/config.ts](frontend/src/config.ts)

## Setup And Run
- Backend install:
  - `pip install -r backend/requirements.txt`
- Backend run (dev):
  - `cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- Frontend install:
  - `cd frontend && npm install`
- Frontend run (dev):
  - `cd frontend && npm run dev`
- Frontend quality/build checks:
  - `cd frontend && npm run lint`
  - `cd frontend && npm run build`

## Environment
- Backend env template: [backend/.env.example](backend/.env.example)
- Frontend env template: [frontend/.env.example](frontend/.env.example)
- Local frontend API should point to backend:
  - `VITE_API_URL=http://localhost:8000`
- CORS must include frontend dev origin(s) in backend env (`CORS_ORIGINS`).

## Architecture Boundaries
- Frontend responsibilities:
  - Detect pose client-side using MediaPipe in [frontend/src/pose/PoseDetector.ts](frontend/src/pose/PoseDetector.ts).
  - Stream landmarks via WebSocket in [frontend/src/hooks/usePoseStream.ts](frontend/src/hooks/usePoseStream.ts).
  - Coordinate video source and frame processing in [frontend/src/hooks/useVideoProcessor.ts](frontend/src/hooks/useVideoProcessor.ts).
- Backend responsibilities:
  - Accept pose stream and return feedback in [backend/api/routes.py](backend/api/routes.py).
  - Manage session/exercise state transitions in [backend/state_machine/manager.py](backend/state_machine/manager.py).
  - Classify exercise and run exercise modules in [backend/exercises/classifier.py](backend/exercises/classifier.py).
  - Exercise plugin interface and shared math in [backend/exercises/base.py](backend/exercises/base.py).

## Exercise Module Conventions
- Add new exercises under [backend/exercises/](backend/exercises/).
- Follow the `BaseExercise` contract in [backend/exercises/base.py](backend/exercises/base.py).
- Keep exercise-specific thresholds in the module itself (see [backend/exercises/squat.py](backend/exercises/squat.py)).
- Ensure returned result contains clear violations and actionable corrections.

## Realtime And Upload Contracts
- WebSocket endpoint and payload contract are documented in [README.md](README.md).
- Chunked upload flow lives in [backend/api/upload.py](backend/api/upload.py):
  - initialize -> upload chunks -> complete

## Pitfalls To Avoid
- Do not call pose detection before async initialization completes in [frontend/src/pose/PoseDetector.ts](frontend/src/pose/PoseDetector.ts).
- Preserve stable client session identity when changing stream logic in [frontend/src/hooks/usePoseStream.ts](frontend/src/hooks/usePoseStream.ts).
- Avoid breaking hysteresis-based rep counting in [backend/utils/rep_counter.py](backend/utils/rep_counter.py).
- Keep smoothing and outlier handling compatible with current joint-angle logic in [backend/utils/smoothing.py](backend/utils/smoothing.py).

## Change Validation
- For frontend-only changes: run lint and build from [frontend/package.json](frontend/package.json).
- For backend changes: start API and verify health/root endpoints and a WebSocket roundtrip.
- If touching protocol fields, update both backend response model and frontend consumers.
