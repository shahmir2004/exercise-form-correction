# 🏋️ Exercise Form Detection & Correction

> Real-time AI-powered exercise form analysis using computer vision

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2-blue.svg)](https://typescriptlang.org)
[![React](https://img.shields.io/badge/React-18.2-61DAFB.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks--Vision-orange.svg)](https://mediapipe.dev)

## 📋 Overview

This application uses **MediaPipe Pose Landmarker** to detect human body poses in real-time and provides intelligent feedback on exercise form. It can identify different exercises, count repetitions, and highlight form violations to help users exercise safely and effectively.

### ✨ Key Features

- **🎯 Real-time Pose Detection** — Client-side ML using MediaPipe Tasks Vision
- **🏃 Exercise Recognition** — Automatically detects Squats, Push-ups, Bicep Curls (standing/seated), and Alternate Curls
- **🔢 Rep Counting (Hysteresis + State Machine)** — Stable counting that resists jitter and phase flicker
- **📈 Rep Quality Scoring** — Tracks rep quality signals (tempo/range/symmetry)
- **⚠️ Form Correction** — Visual and text feedback on form violations with M-of-N aggregation to suppress flicker
- **🎨 Joint Highlighting** — Color-coded joint visualization (green/yellow/red)
- **📡 Kalman-filtered Landmarks** — 3D constant-velocity Kalman filter per joint; visibility-weighted measurement noise
- **🧠 HMM Exercise Classifier** — Log-space Hidden Markov Model replaces heuristic frame counting for smoother, more robust state transitions
- **📐 3D Feature Extraction** — Hip-relative, torso-normalized joint coordinates; view estimation (frontal/profile/three-quarter)
- **🔌 WebSocket Streaming** — Low-latency real-time communication with 60 fps rate limiting
- **📹 Video Upload** — Analyze pre-recorded workout videos
- **📷 Live Camera Mode** — Standalone frontend supports real-time webcam analysis

---

## 🌐 Live Deployment

- **Frontend (Vercel):** `https://frontend-beta-livid-70.vercel.app`
- **Backend (Render):** `https://exercise-form-backend.onrender.com`

### WebSocket (Production)

```
wss://exercise-form-backend.onrender.com/api/ws/pose/{client_id}
```

`client_id` can be any unique string (e.g. `device-123`, `user-abc`, a UUID).

### Frontend Config

Set `VITE_API_URL=https://exercise-form-backend.onrender.com` in your `.env`.

---

## 🏗️ Architecture

### Probabilistic Form-Checking Pipeline

Every WebSocket frame passes through a 6-stage pipeline before a response is sent:

```
Raw landmarks (33 × {x,y,z,vis})
        │
        ▼
┌───────────────────┐
│  InputValidator   │  Checks shape, finite values, visibility stats
└────────┬──────────┘
         │ ValidatedFrame
         ▼
┌───────────────────┐
│  KalmanPoseTracker│  33 independent 6-state filters (pos+vel per axis)
│  (per-joint)      │  R = R_base / max(visibility, 0.05)
└────────┬──────────┘
         │ smoothed xyz (33×3) + uncertainty (33,)
         ▼
┌───────────────────┐
│  FeatureExtractor │  Hip-relative coords · torso-length normalization
│                   │  8 joint angles · view estimation · horizontal check
└────────┬──────────┘
         │ BodyFrame
         ▼
┌───────────────────┐
│  ExerciseHMM      │  5-state log-space forward algorithm
│                   │  Emission: 6 Gaussian features per state
│                   │  Transitions: 0.95 self-loop · 0.02 → IDLE
└────────┬──────────┘
         │ HMMResult (posterior, most_likely_state, exercise_confidence)
         ▼
┌───────────────────┐
│  ExerciseModule   │  HysteresisRepCounter per exercise
│  (Squat/Pushup/   │  Form checks dispatched per detected exercise
│   BicepCurl/etc.) │
└────────┬──────────┘
         │ violations (raw)
         ▼
┌───────────────────┐
│  FormEvaluator    │  M-of-N sliding window per violation code
│  ViolationAgg     │  + cooldown suppression (no repeat within N frames)
└────────┬──────────┘
         │ stable violations
         ▼
┌───────────────────┐
│ ConfidenceComposer│  joint_conf = exp(−uncertainty × 20)
│                   │  exercise-specific joint importance weights
│                   │  partial-body / off-screen penalties
└────────┬──────────┘
         │ FormManagerState
         ▼
    JSON WebSocket response
```

### Session State Machine

```
IDLE  ──────────────────────────────▶  SCANNING
 ▲   (person detected, HMM scanning)       │
 │                                          │ (HMM posterior ≥ 0.7 for non-idle)
 │                                          ▼
 └──────────────────────────────────  ACTIVE
         (confidence drops < 0.3)    (reps + form checks)
```

---

## 📁 Project Structure

```
exercise-form-correction/
│
├── 📂 frontend/                    # React + TypeScript + Vite (standalone test UI)
│   └── src/
│       ├── components/             # ExerciseDisplay, VideoCanvas, VideoUpload, ControlPanel
│       ├── hooks/                  # usePoseStream, useVideoProcessor
│       ├── pose/                   # MediaPipe wrapper + MotionBuffer
│       └── video/                  # VideoSource abstraction (file / camera)
│
├── 📂 backend/
│   ├── main.py                     # FastAPI app, CORS, router mounting
│   ├── requirements.txt
│   ├── pyproject.toml              # pytest configuration
│   │
│   ├── 📂 pipeline/                # ★ NEW — Probabilistic form engine
│   │   ├── validator.py            # InputValidator — shape + finite value checks
│   │   ├── kalman.py               # KalmanPoseTracker — 33 × 6-state Kalman filters
│   │   ├── features.py             # FeatureExtractor — BodyFrame, angles, view estimation
│   │   ├── hmm.py                  # ExerciseHMM — log-space 5-state HMM
│   │   ├── form_evaluator.py       # FormEvaluator + ViolationAggregator (M-of-N)
│   │   ├── confidence.py           # ConfidenceComposer — joint importance × Kalman uncertainty
│   │   └── rep_counter.py          # HysteresisRepCounter (canonical copy)
│   │
│   ├── 📂 api/
│   │   ├── routes.py               # WebSocket endpoint + health + reset
│   │   └── upload.py               # Chunked video upload endpoints
│   │
│   ├── 📂 exercises/               # Exercise modules (pluggable)
│   │   ├── base.py                 # BaseExercise ABC, ExerciseResult, calculate_angle
│   │   ├── classifier.py           # Legacy classifier (kept for hot-swap detection)
│   │   ├── squat.py                # SquatModule
│   │   ├── pushup.py               # PushupModule
│   │   └── bicep_curl.py           # BicepCurlModule + AlternateBicepCurlModule
│   │
│   ├── 📂 state_machine/
│   │   └── manager.py              # FormManager — wires all pipeline stages
│   │
│   ├── 📂 config/
│   │   └── settings.py             # All tunable parameters (env-overridable)
│   │
│   ├── 📂 utils/
│   │   ├── rep_counter.py          # Re-export shim → pipeline/rep_counter.py
│   │   └── smoothing.py            # Legacy EMA smoother (kept for compat)
│   │
│   └── 📂 tests/                   # ★ NEW — pytest test suite (48 tests)
│       ├── conftest.py             # Shared fixtures
│       ├── test_validator.py       # 13 tests
│       ├── test_kalman.py          # 7 tests
│       ├── test_features.py        # 8 tests
│       ├── test_hmm.py             # 7 tests
│       ├── test_form_evaluator.py  # 8 tests (ViolationAggregator + FormEvaluator)
│       └── test_confidence.py      # 5 tests
│
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (frontend only)
- **Git**

### Backend Setup

```bash
git clone https://github.com/shahmir2004/exercise-form-correction.git
cd exercise-form-correction

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r backend/requirements.txt
```

### Run Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API available at `http://localhost:8000` · Swagger docs at `http://localhost:8000/docs`

### Run Tests

```bash
cd backend
pytest
# Expected: 48 passed
```

### Run Frontend (Standalone Test UI)

```bash
cd frontend
npm install
npm run dev
# http://localhost:3000
```

---

## 📊 Supported Exercises

| Exercise | Detection | Rep Counting | Form Checks |
|----------|-----------|--------------|-------------|
| **Squat** | ✅ | ✅ | Knee valgus, depth, back angle, uneven knees |
| **Push-up** | ✅ | ✅ | Hip sag, hip pike, elbow flare, depth, uneven arms |
| **Bicep Curl (Standing/Seated)** | ✅ | ✅ | Elbow drift, body swing, incomplete curl, uneven arms |
| **Alternate Bicep Curl** | ✅ | ✅ | Both-arms-curling, resting arm extension, arm imbalance |
| *Lunge* | 🔜 Planned | — | — |
| *Deadlift* | 🔜 Planned | — | — |

---

## 🔌 API Reference

### WebSocket Endpoint

```
WS /api/ws/pose/{client_id}
```

**Send (Client → Server) — every frame:**
```json
{
  "landmarks": [
    {"x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.99},
    "... 33 landmarks total ..."
  ],
  "timestamp": 1234567890.123
}
```

**Receive (Server → Client) — per frame:**
```json
{
  "state": "active",
  "current_exercise": "squat",
  "exercise_display": "Squat",
  "rep_count": 3,
  "rep_phase": "down",
  "is_rep_valid": true,
  "violations": ["Left knee caving inward"],
  "corrections": ["Push your left knee outward"],
  "correction_message": "Push your left knee outward",
  "joint_colors": {"left_knee": "red", "right_knee": "green"},
  "confidence": 0.88,
  "exercise_confidence": 0.91,
  "form_confidence": 0.88,
  "signal_quality": "good"
}
```

**New fields in this release:**

| Field | Type | Description |
|-------|------|-------------|
| `exercise_confidence` | float 0–1 | HMM posterior for the current exercise |
| `form_confidence` | float 0–1 | Kalman-uncertainty-weighted joint quality score |
| `signal_quality` | `"good"` \| `"degraded"` \| `"unreliable"` | Overall pose signal quality |

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | `{"status":"healthy","connections":N}` |
| `POST` | `/api/reset/{client_id}` | Reset session state |
| `POST` | `/api/upload/init` | Init chunked upload |
| `POST` | `/api/upload/chunk/{upload_id}` | Upload chunk |
| `GET` | `/api/upload/status/{upload_id}` | Upload progress |
| `POST` | `/api/upload/complete/{upload_id}` | Assemble final file |

---

## ⚙️ Configuration

All parameters are tunable via environment variables (see `backend/config/settings.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Port (Render injects this) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `KALMAN_PROCESS_NOISE` | `1e-3` | Kalman Q — higher = more responsive |
| `HMM_TRANSITION_SELF_LOOP` | `0.95` | HMM state persistence (higher = slower transitions) |
| `VIOLATION_AGG_M` | `4` | M-of-N: violations needed in a window to emit |
| `VIOLATION_AGG_N` | `6` | M-of-N: window size (frames) |
| `VIOLATION_COOLDOWN_FRAMES` | `15` | Frames before same violation can emit again |
| `MAX_FRAMES_PER_SECOND` | `60` | Rate limit for WebSocket frames |

---

## 🛠️ Tech Stack

### Frontend
- **React 18** · **TypeScript** · **Vite** · **Tailwind CSS**
- **MediaPipe Tasks Vision** — client-side GPU pose detection

### Backend
- **FastAPI** + **Uvicorn** — ASGI web framework
- **NumPy** + **SciPy** — Kalman filter, HMM, angle calculations
- **Pydantic v2** — settings + request validation
- **pytest** — 48-test suite covering every pipeline stage

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/shahmir2004">
        <img src="https://github.com/shahmir2004.png" width="100px;" alt="Muhammad Shahmir Ahmed"/><br />
        <sub><b>Muhammad Shahmir Ahmed</b></sub>
      </a><br />
      <sub>Lead Developer</sub>
    </td>
    <td align="center">
      <a href="https://github.com/ABDULLAHAZHERCH">
        <img src="https://github.com/ABDULLAHAZHERCH.png" width="100px;" alt="Abdullah Azher Chaudry"/><br />
        <sub><b>Abdullah Azher Chaudry</b></sub>
      </a><br />
      <sub>Lead Developer</sub>
    </td>
  </tr>
</table>

**Muhammad Shahmir Ahmed** — [Portfolio](https://shahmir-ahmed.vercel.app) · [GitHub](https://github.com/shahmir2004)

**Abdullah Azher Chaudry** — [Portfolio](https://abdullahch.vercel.app) · [GitHub](https://github.com/ABDULLAHAZHERCH)

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the pose detection model
- [FastAPI](https://fastapi.tiangolo.com/) for the Python web framework
- [React](https://reactjs.org/) for the frontend framework

---

<p align="center">Made with ❤️ for fitness enthusiasts everywhere</p>
