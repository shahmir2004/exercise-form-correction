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

- **🎯 Real-time Pose Detection** — Client-side ML using MediaPipe Tasks Vision (GPU)
- **🧠 ST-GCN Client Classifier** — Browser-side Spatio-Temporal Graph Convolutional Network distinguishes 4 bicep curl variants before any frame reaches the server
- **🏃 Exercise Recognition** — Automatically detects Squats, Push-ups, Bicep Curls (standing/seated), and Alternate Curls
- **🔢 Rep Counting (Hysteresis + State Machine)** — Stable counting that resists jitter and phase flicker
- **📈 Rep Quality Scoring** — Tracks rep quality signals (tempo/range/symmetry)
- **⚠️ Form Correction** — Visual and text feedback on form violations with M-of-N aggregation to suppress flicker
- **🎨 Joint Highlighting** — Color-coded joint visualization (green/yellow/red)
- **📡 Kalman-filtered Landmarks** — 3D constant-velocity Kalman filter per joint; visibility-weighted measurement noise
- **🤝 HMM + k-NN Fusion Classifier** — Two independent classifiers cross-validate each other; agreement boosts confidence, disagreement triggers hysteresis fallback
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

## 🧠 ST-GCN Client-Side Classifier

### What Is It?

The **Spatio-Temporal Graph Convolutional Network (ST-GCN)** is a neural network that runs entirely in the browser — no server round-trip. It watches a 30-frame sliding window of pose landmarks and outputs a probability distribution over 4 bicep curl variants before any data is sent to the backend.

### Why Is It Needed?

The backend HMM classifier works well for distinguishing broad exercise categories (squat vs. push-up vs. curl). But **standing vs. seated** bicep curls and **alternating vs. simultaneous** curls look very similar in hip/elbow angles alone. The ST-GCN learns **temporal motion patterns** — the rhythm and asymmetry of alternating reps — that are invisible to a per-frame HMM.

By sending `client_probs` alongside raw landmarks, the backend fuses both signals for a more reliable exercise variant decision.

### Architecture

```
30 frames × 17 joints × 3 coords
         │
         ▼
  ┌──────────────────────┐
  │  Torso Normalisation │  Hip midpoint → origin; torso length → unit scale
  └──────────┬───────────┘
             │ (30, 17, 3) tensor
             ▼
  ┌──────────────────────┐
  │   Feature Scaler     │  Per-feature z-score: (x − μ) / σ
  │   (stgcn_scaler.json)│  μ, σ computed over synthetic training data
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────────────┐
  │  Graph Conv Layer 1          │
  │  A × X → W₁ + b₁ → ReLU    │  (17, 3) → (17, 64) per frame
  │  Adjacency: degree-norm +    │  A is symmetric, self-loops included
  │  self-loops (stgcn_weights)  │
  └──────────┬───────────────────┘
             │ (30, 17, 64)
             ▼
  ┌──────────────────────────────┐
  │  Graph Conv Layer 2          │
  │  A × H₁ → W₂ + b₂ → ReLU  │  (17, 64) → (17, 64) per frame
  └──────────┬───────────────────┘
             │ (30, 17, 64)
             ▼
  ┌──────────────────────┐
  │  Global Avg Pool     │  Mean over 30 time steps → (17 × 64 = 1088,)
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Fully Connected     │  (1088,) → (4,) logits → Softmax
  └──────────┬───────────┘
             │
             ▼
  { "curl-stand": 0.82, "curl-seat": 0.06,
    "alt-stand":  0.09, "alt-seat":  0.03 }
```

### Key Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `WINDOW` | 30 | Frames required before first inference (~1 second at 30fps) |
| `N_JOINTS` | 17 | Key joints selected from MediaPipe's 33 (shoulders, elbows, wrists, hips, knees, ankles, nose, ears) |
| `COORD_DIM` | 3 | x, y, z per joint |
| `HIDDEN` | 64 | Graph conv channel width |
| `N_CLASSES` | 4 | curl-stand, curl-seat, alt-stand, alt-seat |

### 17 Key Joints

Indices into MediaPipe's 33-landmark output:

```
[11, 12] = Left/Right Shoulder
[13, 14] = Left/Right Elbow
[15, 16] = Left/Right Wrist
[23, 24] = Left/Right Hip
[25, 26] = Left/Right Knee
[27, 28] = Left/Right Ankle
[0]       = Nose
[7,  8]  = Left/Right Ear
[9, 10]  = Left/Right Mouth corner
```

### Torso Normalisation

Before any inference, each frame is translated and scaled so the network is invariant to the user's distance from the camera:

```
origin    = midpoint of left hip (23) and right hip (24)
scale     = distance from hip midpoint to shoulder midpoint
            (torso length, average of left and right)

normalised_x = (raw_x − origin_x) / scale
normalised_y = (raw_y − origin_y) / scale
normalised_z = raw_z / scale
```

This means a tall person standing far away and a short person standing close produce identical input tensors for the same pose.

### Graph Structure

The 17 joints are connected by an adjacency matrix that mirrors human skeletal connectivity. The matrix is:
- **Symmetric** — edges are bidirectional
- **Degree-normalised** — each row sums to 1 (D⁻¹A), so message passing is a weighted average of neighbours
- **Self-loops included** — each joint attends to itself plus its neighbours

### Weight Generation (`tools/generate_stgcn_weights.py`)

Because real labelled per-frame exercise data is scarce, weights are trained on **synthetic data**:

1. **Dataset generation** — For each of 4 classes, 200 sequences of 30 frames are simulated. Each class gets a distinctive motion signature:
   - `curl-stand`: symmetric elbow flexion, vertical torso
   - `curl-seat`: same elbow motion, lowered knee position (seated geometry)
   - `alt-stand`: alternating elbow flexion (left/right out of phase by 15 frames), vertical torso
   - `alt-seat`: alternating motion + seated geometry

2. **Training** — Standard backprop through the two graph-conv layers and FC head. NumPy only (no PyTorch/TensorFlow). 80 epochs, learning rate 2e-2, cross-entropy loss.

3. **Export** — Weights saved as `frontend/public/stgcn_weights.json` (~231KB). Scaler statistics (mean + std per feature) saved as `frontend/public/stgcn_scaler.json`.

To retrain:
```bash
cd tools
python generate_stgcn_weights.py
# Outputs: ../form-checking-backend/frontend/public/stgcn_weights.json
#          ../form-checking-backend/frontend/public/stgcn_scaler.json
```

---

## 🤝 HMM + k-NN Fusion Classifier (Backend)

### Motivation

The previous backend used a single HMM for exercise classification. A single mistuned emission parameter could silently break classification for an entire exercise. The new design uses **two independent classifiers** that cross-validate each other.

### Two Classifiers

**HMM (primary)**
- 5-state log-space forward algorithm: IDLE, SQUAT, PUSHUP, CURL, ALT_CURL
- Features: knee angles, elbow angles, hip height, body horizontal tilt, arm asymmetry, phase difference
- Strength: temporal smoothing — won't flip between exercises mid-rep

**k-NN (safety net)**
- Pose embedding: 66 pairwise distances between all pairs of 12 key joints
- Scale- and rotation-invariant (distances are unitless after torso normalisation)
- Reference library: ~150 representative frames per exercise stored in `backend/data/pose_library/`
- Strength: no training required; works directly from examples

### Fusion Rules (`backend/pipeline/fusion.py`)

```
HMM result + k-NN result
        │
        ▼
┌─────────────────────────────────┐
│  Both agree?                    │ → Use HMM (temporal continuity)
│  Weighted score: 0.6×HMM + 0.4×kNN
├─────────────────────────────────┤
│  Both confident > 0.7 but       │
│  disagree?                      │ → Prefer HMM (continuity bias)
├─────────────────────────────────┤
│  Confidence gap > 0.3?          │ → Use the more confident one
├─────────────────────────────────┤
│  HMM alone > 0.6?               │ → Use HMM
├─────────────────────────────────┤
│  Otherwise                      │ → Emit idle (uncertain)
└─────────────────────────────────┘
```

The k-NN acts as a **sanity check** — if it strongly disagrees with HMM, the fusion will not blindly trust the HMM's temporal momentum.

---

## 🔗 End-to-End Data Flow

```
User performs exercise in front of camera
              │
              ▼
  MediaPipe PoseLandmarker (browser, GPU)
  Outputs 33 landmarks × {x, y, z, visibility}
              │
              ├──────────────────────────────────┐
              │                                  │
              ▼                                  ▼
  Skeleton overlay drawn on canvas    ST-GCN sliding window buffer
  (joint_colors applied from          Accumulates 30 frames of
   last server response)              17-joint torso-normalised poses
                                               │
                                               │ (every frame after window fills)
                                               ▼
                                      STGCNClassifier.infer()
                                      → { "curl-stand": 0.82, ... }
                                      stored in clientProbsRef (no re-render)
              │
              ▼
  usePoseStream.sendLandmarks(landmarks, timestamp, clientProbs)
              │
              ▼
  WebSocket frame → FastAPI backend
  {
    "landmarks": [...33 landmarks...],
    "timestamp": 1234567.89,
    "client_probs": { "curl-stand": 0.82, "curl-seat": 0.06, ... }  ← optional
  }
              │
              ▼
  FormManager processes frame:
  1. InputValidator — checks shape + finite values
  2. KalmanPoseTracker — smooths 33 joints (visibility-weighted noise)
  3. FeatureExtractor — torso-normalised coords + 8 joint angles + view
  4. ExerciseHMM — log-space forward algorithm → posterior per exercise
  5. k-NN Classifier — pairwise-distance embedding → nearest-neighbour lookup
  6. ClassifierFusion — fuses HMM + k-NN → final exercise decision
  7. ExerciseModule (Squat/Pushup/BicepCurl) — rep counter + form checks
  8. ViolationAggregator — M-of-N sliding window, cooldown suppression
  9. ConfidenceComposer — joint importance × Kalman uncertainty → score
              │
              ▼
  WebSocket response → browser
  {
    "state": "active",
    "current_exercise": "bicep_curl",
    "rep_count": 5,
    "violations": ["Left elbow drifting forward"],
    "joint_colors": { "left_elbow": "red", ... },
    "confidence": 0.91,
    ...
  }
              │
              ▼
  App.tsx updates UI: rep counter, correction text, joint colors on canvas
```

---

## 🏗️ Architecture

### Probabilistic Form-Checking Pipeline

Every WebSocket frame passes through a 9-stage pipeline before a response is sent:

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
         │
         ▼
┌───────────────────┐
│  k-NN Classifier  │  66-feature pairwise-distance pose embedding
│                   │  Nearest-neighbour lookup in reference library
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  ClassifierFusion │  HMM + k-NN cross-validation with hysteresis rules
└────────┬──────────┘
         │ FusionResult (exercise, confidence, source)
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
 │                                          │ (fusion confidence ≥ 0.7)
 │                                          ▼
 └──────────────────────────────────  ACTIVE
         (confidence drops < 0.3)    (reps + form checks)
```

---

## 📁 Project Structure

```
exercise-form-correction/
│
├── 📂 tools/                           # ★ NEW — Offline model training
│   ├── generate_stgcn_weights.py       # Synthetic dataset + ST-GCN training + weight export
│   └── test_generate_weights.py        # pytest: 8 tests for training pipeline
│
├── 📂 frontend/                        # React + TypeScript + Vite (standalone test UI)
│   ├── public/
│   │   ├── stgcn_weights.json          # ★ NEW — Trained graph-conv + FC weights (~231KB)
│   │   └── stgcn_scaler.json           # ★ NEW — Per-feature mean/std for normalisation
│   └── src/
│       ├── components/                 # ExerciseDisplay, VideoCanvas, VideoUpload, ControlPanel
│       ├── hooks/
│       │   ├── usePoseStream.ts        # ★ UPDATED — sendLandmarks now accepts client_probs
│       │   └── useVideoProcessor.ts
│       ├── pose/
│       │   ├── STGCNClassifier.ts      # ★ NEW — Browser ST-GCN inference (Float32Array)
│       │   ├── PoseDetector.ts         # MediaPipe wrapper
│       │   ├── MotionBuffer.ts         # Circular sliding window
│       │   └── index.ts               # ★ UPDATED — exports STGCNClassifier + constants
│       ├── App.tsx                     # ★ UPDATED — ST-GCN pipeline wired into frame loop
│       └── video/                      # VideoSource abstraction (file / camera)
│
├── 📂 backend/
│   ├── main.py                         # FastAPI app, CORS, router mounting
│   ├── requirements.txt
│   │
│   ├── 📂 pipeline/                    # Probabilistic form engine
│   │   ├── validator.py                # InputValidator
│   │   ├── kalman.py                   # KalmanPoseTracker
│   │   ├── features.py                 # FeatureExtractor + BodyFrame
│   │   ├── hmm.py                      # ExerciseHMM (5-state log-space)
│   │   ├── knn_classifier.py           # ★ NEW — k-NN pose embedding classifier
│   │   ├── pose_embedder.py            # ★ NEW — Pairwise-distance pose embedding (66 features)
│   │   ├── fusion.py                   # ★ NEW — ClassifierFusion (HMM + k-NN)
│   │   ├── form_evaluator.py           # ViolationAggregator (M-of-N)
│   │   ├── confidence.py               # ConfidenceComposer
│   │   └── rep_counter.py              # HysteresisRepCounter
│   │
│   ├── 📂 api/
│   │   ├── routes.py                   # ★ UPDATED — reads client_probs from WS frames
│   │   ├── pose_library.py             # ★ NEW — REST endpoints for k-NN reference library
│   │   └── upload.py                   # Chunked video upload endpoints
│   │
│   ├── 📂 data/
│   │   └── pose_library/               # ★ NEW — Reference pose JSON files for k-NN
│   │       ├── bicep_curl.json
│   │       └── alternate_bicep_curl.json
│   │
│   ├── 📂 state_machine/
│   │   └── manager.py                  # ★ UPDATED — fusion pipeline integrated
│   │
│   ├── 📂 exercises/                   # Exercise modules (pluggable)
│   │   ├── base.py
│   │   ├── classifier.py               # Legacy classifier (kept for ExerciseType enum)
│   │   ├── squat.py
│   │   ├── pushup.py
│   │   └── bicep_curl.py
│   │
│   ├── 📂 config/
│   │   └── settings.py                 # ★ UPDATED — fusion + k-NN parameters added
│   │
│   └── 📂 tests/                       # pytest test suite
│       └── ...
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

# Python ST-GCN weight generator tests:
cd tools
pytest test_generate_weights.py -v
# Expected: 8 passed
```

### Run Frontend (Standalone Test UI)

```bash
cd frontend
npm install
npm run dev
# http://localhost:3000

# Frontend unit tests:
npm test
# Expected: 10 passed (STGCNClassifier × 7, buildPoseMessage × 3)
```

### Retrain ST-GCN Weights (optional)

```bash
cd tools
python generate_stgcn_weights.py
# Rewrites frontend/public/stgcn_weights.json and stgcn_scaler.json
```

---

## 📊 Supported Exercises

| Exercise | Detection | Rep Counting | Form Checks |
|----------|-----------|--------------|-------------|
| **Squat** | ✅ | ✅ | Knee valgus, depth, back angle, uneven knees |
| **Push-up** | ✅ | ✅ | Hip sag, hip pike, elbow flare, depth, uneven arms |
| **Bicep Curl (Standing)** | ✅ | ✅ | Elbow drift, body swing, incomplete curl, uneven arms |
| **Bicep Curl (Seated)** | ✅ | ✅ | Same as standing; ST-GCN distinguishes variant |
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
  "timestamp": 1234567890.123,
  "client_probs": {
    "curl-stand": 0.82,
    "curl-seat":  0.06,
    "alt-stand":  0.09,
    "alt-seat":   0.03
  }
}
```

`client_probs` is **optional**. If omitted, the backend classifies using HMM + k-NN alone. If present, the probabilities refine curl variants only; squat and push-up detection remain guarded by backend pose rules.

**Receive (Server → Client) — per frame:**
```json
{
  "state": "active",
  "current_exercise": "bicep_curl",
  "exercise_display": "Bicep Curl",
  "rep_count": 3,
  "rep_phase": "down",
  "is_rep_valid": true,
  "violations": ["Left elbow drifting forward"],
  "corrections": ["Keep your left elbow pinned to your side"],
  "correction_message": "Keep your left elbow pinned to your side",
  "joint_colors": {"left_elbow": "red", "right_elbow": "green"},
  "confidence": 0.88,
  "exercise_confidence": 0.91,
  "form_confidence": 0.88,
  "signal_quality": "good",
  "exercise_variant": "curl-stand",
  "exercise_source": "external_variant"
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `state` | `"idle"` \| `"scanning"` \| `"active"` | Session state machine state |
| `current_exercise` | string \| null | Detected exercise key |
| `rep_count` | int | Total completed reps this session |
| `rep_phase` | string | Current phase within a rep (`up`/`down`/`hold`/etc.) |
| `violations` | string[] | Active form issues |
| `joint_colors` | object | Per-joint color (`"green"`, `"yellow"`, `"red"`) |
| `exercise_confidence` | float 0–1 | Fusion classifier confidence |
| `form_confidence` | float 0–1 | Kalman-uncertainty-weighted joint quality |
| `signal_quality` | string | Overall pose signal quality |
| `exercise_variant` | string \| null | Variant label, e.g. `curl-stand` or `alt-seat` |
| `exercise_source` | string | Decision source, e.g. `rule_gate`, `fusion_agree`, or `external_variant` |

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | `{"status":"healthy","connections":N}` |
| `POST` | `/api/reset/{client_id}` | Reset session state |
| `GET` | `/api/pose-library` | List k-NN reference libraries |
| `GET` | `/api/pose-library/{exercise}` | Download a reference library |
| `POST` | `/api/pose-library/record` | Save captured pose frames as k-NN embeddings |
| `DELETE` | `/api/pose-library/{exercise}` | Delete a reference library |
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
| `HMM_TRANSITION_SELF_LOOP` | `0.95` | HMM state persistence |
| `FUSION_HMM_WEIGHT` | `0.6` | HMM weight in fusion when classifiers agree |
| `FUSION_KNN_WEIGHT` | `0.4` | k-NN weight in fusion when classifiers agree |
| `VIOLATION_AGG_M` | `4` | M-of-N: violations needed in a window to emit |
| `VIOLATION_AGG_N` | `6` | M-of-N: window size (frames) |
| `VIOLATION_COOLDOWN_FRAMES` | `15` | Frames before same violation can emit again |
| `MAX_FRAMES_PER_SECOND` | `60` | Rate limit for WebSocket frames |
| `MIN_CLASS_LIBRARY_EMBEDDINGS` | `1` | Minimum embeddings required for a k-NN class to vote |
| `PUSHUP_HORIZONTAL_MIN_CONFIDENCE` | `0.78` | Rule-gate confidence for horizontal push-up detection |
| `SQUAT_RULE_GATE_CONFIDENCE` | `0.72` | Rule-gate confidence for squat detection |
| `CURL_VARIANT_OVERRIDE_CONFIDENCE` | `0.7` | Minimum ST-GCN confidence to refine curl variants |

---

## 🛠️ Tech Stack

### Frontend
- **React 18** · **TypeScript** · **Vite** · **Tailwind CSS**
- **MediaPipe Tasks Vision** — client-side GPU pose detection
- **Vitest** — unit tests for ST-GCN classifier and WebSocket message builder

### Backend
- **FastAPI** + **Uvicorn** — ASGI web framework
- **NumPy** + **SciPy** — Kalman filter, HMM, angle calculations
- **Pydantic v2** — settings + request validation
- **pytest** — test suite covering every pipeline stage

### Model Training
- **NumPy-only backprop** — no PyTorch or TensorFlow dependency
- **Synthetic dataset** — 200 sequences × 4 classes × 30 frames, generated offline
- **Offline training** — weights are pre-trained and bundled as static JSON

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
