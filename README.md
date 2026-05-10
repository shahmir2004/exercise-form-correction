# Exercise Form Correction Backend

FastAPI service for real-time exercise detection, rep counting, and form feedback. This repo also includes a standalone Vite demo frontend used for backend integration testing, but the production app client is the separate Gymi repo.

## Current Status

- Production backend: `https://exercise-form-backend.onrender.com`
- Production WebSocket: `wss://exercise-form-backend.onrender.com/api/ws/pose/{client_id}`
- Local backend: `http://localhost:8000`
- Local WebSocket: `ws://localhost:8000/api/ws/pose/{client_id}`
- Render blueprint: [render.yaml](render.yaml)

`client_id` must be 80 characters or fewer and may contain letters, numbers, `_`, `.`, `:`, or `-`.

## What It Does

- Accepts MediaPipe-style 33-landmark pose frames over WebSocket.
- Validates and smooths landmarks with a per-joint Kalman tracker.
- Extracts hip-relative, torso-normalized 3D features and joint angles.
- Classifies exercise state with an HMM plus rule-based safety gates.
- Detects stationary users for clearer UX states.
- Switches exercises mid-session using confidence, timing, and rep-phase gates.
- Counts reps with a hysteresis counter that rejects jitter and shallow partials.
- Runs exercise-specific form checks and stable violation aggregation.
- Returns rep phase, correction text, joint colors, confidence, signal quality, and camera-view metadata.
- Supports chunked video uploads for recorded workout files.

## Supported Exercises

| Exercise | Label | Variants | Current Support |
| --- | --- | --- | --- |
| Squat | `squat` | none | Detection, reps, form checks |
| Push-up | `pushup` | none | Detection, reps, form checks |
| Bicep Curl | `bicep_curl` | `curl-stand`, `curl-seat` | Detection, reps, form checks |
| Alternate Bicep Curl | `alternate_bicep_curl` | `alt-stand`, `alt-seat` | Detection, per-arm reps, form checks |

The canonical exercise list lives in [backend/exercises/registry.py](backend/exercises/registry.py).

## Runtime Pipeline

Every WebSocket frame flows through the current backend pipeline:

```text
landmarks
  -> InputValidator
  -> KalmanPoseTracker
  -> FeatureExtractor
  -> ExerciseHMM
  -> rule-based safety gate
  -> MotionDetector
  -> exercise switching hysteresis
  -> active ExerciseModule
  -> HysteresisRepCounter
  -> FormEvaluator
  -> ConfidenceComposer
  -> WebSocket response
```

Important implementation notes:

- The current backend intentionally uses a single HMM classifier with rule-gate overrides. Deleted k-NN, fusion, and pose-embedder modules are covered by regression tests so they do not reappear accidentally.
- Push-up and squat rule gates can override a weak curl candidate when body orientation or knee/hip geometry is clearly stronger.
- Alternating curls use elbow asymmetry plus left/right elbow velocity phase to survive the crossover frame where both arms briefly look similar.
- Stationary detection is a UX state. It does not directly count or reject reps.
- Rep phase labels are lift-semantic: for bicep curls, `concentric` means curling up even though elbow angle is decreasing.
- The route currently reads `landmarks`, `timestamp`, and optional `camera_view`. Extra client keys are ignored by the backend route.

## API

### WebSocket Pose Stream

```text
WS /api/ws/pose/{client_id}
```

Client frame:

```json
{
  "landmarks": [
    { "x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.99 }
  ],
  "timestamp": 1778390000000,
  "camera_view": "auto"
}
```

`landmarks` should contain 33 MediaPipe pose landmarks. `camera_view` is optional and accepts `auto`, `front`, `side`, or `three_quarter`. If omitted or set to `auto`, the backend returns its estimated view.

Server response:

```json
{
  "state": "active",
  "current_exercise": "bicep_curl",
  "exercise_display": "Activity: Bicep Curl",
  "rep_count": 3,
  "rep_phase": "concentric",
  "phase_display": "Curling up",
  "is_rep_valid": true,
  "violations": ["Left elbow drifting forward"],
  "corrections": ["Keep your left elbow pinned to your side"],
  "correction_message": "Keep your left elbow pinned to your side",
  "joint_colors": { "left_elbow": "red" },
  "confidence": 0.88,
  "is_stationary": false,
  "timestamp": 1778390000000,
  "exercise_confidence": 0.91,
  "form_confidence": 0.88,
  "signal_quality": "good",
  "exercise_variant": "bicep_curl",
  "exercise_source": "hmm",
  "camera_view": "frontal"
}
```

Response states are `idle`, `stationary`, `scanning`, and `active`. Rep phases are `idle`, `setup`, `eccentric`, `concentric`, and `hold`.

### REST Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service metadata |
| `GET` | `/api/health` | Health, active WebSocket count, supported exercises |
| `GET` | `/api/exercises` | Current supported exercise contract |
| `POST` | `/api/reset/{client_id}` | Reset a client's detection session |
| `POST` | `/api/upload/init` | Start chunked video upload |
| `POST` | `/api/upload/chunk/{upload_id}` | Upload one video chunk |
| `GET` | `/api/upload/status/{upload_id}` | Check upload progress |
| `POST` | `/api/upload/complete/{upload_id}` | Assemble uploaded chunks |
| `DELETE` | `/api/upload/{upload_id}` | Cancel an upload session |
| `GET` | `/api/upload/files` | List uploaded video files |

Upload files are limited by `MAX_FILE_SIZE`, split by `CHUNK_SIZE`, validated by file extension, and can optionally be checked against a SHA-256 hash.

## Repository Layout

```text
backend/
  api/
    routes.py              WebSocket, health, exercises, reset
    upload.py              Chunked video upload API
  config/
    settings.py            Environment-backed configuration
  exercises/
    base.py                Shared exercise contracts and phase mapping
    squat.py               Squat module
    pushup.py              Push-up module
    bicep_curl.py          Standard and alternate curl modules
    registry.py            Supported exercise metadata
  pipeline/
    validator.py           Input validation
    kalman.py              Landmark smoothing
    features.py            BodyFrame feature extraction
    hmm.py                 HMM classifier
    motion_detector.py     Stationary detection
    rep_counter.py         Hysteresis rep counting
    form_evaluator.py      Stable violation aggregation
    confidence.py          Signal and form confidence
  state_machine/
    manager.py             Main orchestration pipeline
  tests/                   Pytest coverage
  scripts/
    ws_replay_smoke.py     End-to-end WebSocket replay smoke test
  main.py                  FastAPI app entry point
frontend/
  src/                     Standalone Vite test UI
  public/                  Browser-side demo assets
```

## Local Setup

### Backend

```powershell
cd "D:\work\FYP FINAL\form-checking-backend"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend\requirements.txt
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger docs are available at `http://localhost:8000/docs`.

### Standalone Test Frontend

```powershell
cd "D:\work\FYP FINAL\form-checking-backend\frontend"
npm install
npm run dev
```

The standalone frontend defaults to the backend URL configured in [frontend/src/config.ts](frontend/src/config.ts). Gymi is the production client-facing app.

## Configuration

Settings are loaded from environment variables by [backend/config/settings.py](backend/config/settings.py).

| Variable | Default | Notes |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Render injects this in production |
| `DEBUG` | `False` | Also accepts `dev`, `debug`, `prod`, `production`, `release` |
| `CORS_ORIGINS` | local dev + Vercel defaults | Comma-separated origins or `*` |
| `CORS_ORIGIN_REGEX` | unset | Useful for Vercel preview deployments |
| `MAX_FRAMES_PER_SECOND` | `60` | Per-client WebSocket frame limit |
| `MAX_CLIENT_ID_LENGTH` | `80` | WebSocket client id guard |
| `STATIONARY_WINDOW_FRAMES` | `30` | Rolling stationary-detection window |
| `STATIONARY_THRESHOLD` | `0.015` | Max normalized landmark motion for stationary |
| `DETECTION_DEBUG_LOG` | `False` | Enables per-frame debug logs |
| `DETECTION_STICKY_FLOOR` | `0.3` | Keeps active exercise visible through brief dips |
| `MIN_CONFIDENCE_FOR_REPS` | `0.4` | Start threshold for rep counting |
| `EXERCISE_SWITCH_CONFIDENCE` | `0.6` | Switch threshold once an exercise is active |
| `EXERCISE_SWITCH_MIN_SECONDS` | `0.5` | Minimum candidate duration before switch |
| `EXERCISE_SWITCH_MIN_FRAMES` | `6` | Minimum candidate frames before switch |
| `EXERCISE_SWITCH_IDLE_SECONDS` | `0.75` | Safe between-rep switch timing |
| `EXERCISE_DROP_SECONDS` | `1.0` | Different-exercise persistence for forced switch |
| `SQUAT_RULE_GATE_CONFIDENCE` | `0.72` | Squat rule-gate confidence |
| `PUSHUP_HORIZONTAL_MIN_CONFIDENCE` | `0.78` | Push-up rule-gate confidence |
| `VIOLATION_AGG_M` | `4` | Required repeated violations |
| `VIOLATION_AGG_N` | `6` | Violation aggregation window |
| `VIOLATION_COOLDOWN_FRAMES` | `15` | Repeat violation cooldown |
| `UPLOAD_DIR` | `./uploads` | Final video upload directory |
| `CHUNK_DIR` | `./uploads/chunks` | Temporary chunk directory |
| `MAX_FILE_SIZE` | `5GB` | Upload size guard |
| `CHUNK_SIZE` | `5MB` | Upload chunk size |
| `SUPABASE_ENABLED` | `False` | Optional, currently disabled for MVP |

The Render blueprint currently deploys the `backend/` folder with Python 3.11 and starts:

```powershell
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Testing

Backend tests:

```powershell
cd "D:\work\FYP FINAL\form-checking-backend\backend"
pytest
```

Current collection: 69 tests across validator, Kalman, feature extraction, HMM, rule gates, form evaluation, rep counting, partial-body curl handling, uploads, and production contracts.

Standalone frontend tests:

```powershell
cd "D:\work\FYP FINAL\form-checking-backend\frontend"
npm test
```

Current collection: 10 Vitest tests.

Synthetic backend smoke test:

```powershell
cd "D:\work\FYP FINAL\form-checking-backend\backend"
python smoke_test.py
```

End-to-end WebSocket replay against a running local server:

```powershell
cd "D:\work\FYP FINAL\form-checking-backend\backend"
python scripts\ws_replay_smoke.py --url ws://127.0.0.1:8000/api/ws/pose/smoke
```

## Production Client Integration

Gymi should configure:

```env
NEXT_PUBLIC_FORM_COACH_URL=https://exercise-form-backend.onrender.com
```

The Gymi hook derives `wss://` from this value, connects to `/api/ws/pose/{client_id}`, sends landmarks plus `camera_view`, and displays the backend response.

For new clients:

- Send exactly 33 MediaPipe pose landmarks.
- Keep timestamps monotonic where possible.
- Send no faster than the configured backend frame limit.
- Use `/api/exercises` to avoid hard-coding exercise labels.
- Treat `confidence` as a legacy alias for `form_confidence`.
- Prefer `phase_display` for user-facing text and `rep_phase` for program logic.

## Authors

- Muhammad Shahmir Ahmed - [GitHub](https://github.com/shahmir2004)
- Abdullah Azher Chaudry - [GitHub](https://github.com/ABDULLAHAZHERCH)

## License

MIT. See [LICENSE](LICENSE).
