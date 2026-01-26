# 🏋️ Exercise Form Detection & Correction

> Real-time AI-powered exercise form analysis using computer vision

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2-blue.svg)](https://typescriptlang.org)
[![React](https://img.shields.io/badge/React-18.2-61DAFB.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688.svg)](https://fastapi.tiangolo.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks--Vision-orange.svg)](https://mediapipe.dev)

## 📋 Overview

This application uses **MediaPipe Pose Landmarker** to detect human body poses in real-time and provides intelligent feedback on exercise form. It can identify different exercises, count repetitions, and highlight form violations to help users exercise safely and effectively.

### ✨ Key Features

- **🎯 Real-time Pose Detection** - Client-side ML using MediaPipe Tasks Vision
- **🏃 Exercise Recognition** - Automatically detects Squats, Push-ups, Bicep Curls (standing/seated), and Alternate Curls
- **🔢 Rep Counting (Hysteresis + State Machine)** - More stable counting that resists jitter and phase flicker
- **📈 Rep Quality Scoring** - Tracks rep quality signals (tempo/range/symmetry) for better coaching feedback
- **⚠️ Form Correction** - Visual and text feedback on form violations
- **🎨 Joint Highlighting** - Color-coded joint visualization (green/yellow/red)
- **🧠 Pose Smoothing (EMA + Outlier Rejection)** - Landmark smoothing for steadier angles and fewer false triggers
- **📹 Video Upload** - Analyze pre-recorded workout videos
- **🔌 WebSocket Streaming** - Low-latency real-time communication

---

## 🌐 Live Deployment

- Frontend (Vercel): `https://frontend-beta-livid-70.vercel.app`
- Backend (Render): `https://exercise-form-backend.onrender.com`

### WebSocket (Production)

- Endpoint: `wss://exercise-form-backend.onrender.com/api/ws/pose/{client_id}`
- `client_id` can be any unique identifier (e.g. `device-123`, `user-abc`, a UUID)

### Frontend Config

- Set `VITE_API_URL=https://exercise-form-backend.onrender.com`

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Video Source │──│ PoseDetector │──│ Canvas Overlay       │   │
│  │ (File/Camera)│  │ (MediaPipe)  │  │ (Skeleton Drawing)   │   │
│  └──────────────┘  └──────┬───────┘  └──────────────────────┘   │
│                           │ 33 Landmarks                         │
│                           ▼                                      │
│                    ┌─────────────┐                               │
│                    │  WebSocket  │                               │
│                    │   Client    │                               │
│                    └──────┬──────┘                               │
└───────────────────────────│─────────────────────────────────────┘
                            │ JSON (landmarks + timestamp)
                            ▼
┌───────────────────────────│─────────────────────────────────────┐
│                    ┌──────┴──────┐                               │
│                    │  WebSocket  │         BACKEND (FastAPI)     │
│                    │   Server    │                               │
│                    └──────┬──────┘                               │
│                           │                                      │
│  ┌────────────────────────┼────────────────────────────────┐     │
│  │              State Machine Manager                       │     │
│  │  ┌─────────┐    ┌──────────┐    ┌─────────────────┐     │     │
│  │  │  IDLE   │───▶│ SCANNING │───▶│     ACTIVE      │     │     │
│  │  │         │◀───│          │◀───│ (Counting Reps) │     │     │
│  │  └─────────┘    └──────────┘    └─────────────────┘     │     │
│  └─────────────────────────────────────────────────────────┘     │
│                           │                                      │
│  ┌────────────────────────┼────────────────────────────────┐     │
│  │              Exercise Classifier                         │     │
│  │  Analyzes joint angles to identify exercise type         │     │
│  └────────────────────────┼────────────────────────────────┘     │
│                           │                                      │
│  ┌────────────────────────┴────────────────────────────────┐     │
│  │              Exercise Modules (Pluggable)                │     │
│  │  ┌─────────┐    ┌──────────┐    ┌─────────────────┐     │     │
│  │  │  Squat  │    │ Push-up  │    │   Bicep Curl    │     │     │
│  │  │ Module  │    │  Module  │    │     Module      │     │     │
│  │  └─────────┘    └──────────┘    └─────────────────┘     │     │
│  └─────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

### Modular Architecture

The project follows a **modular, plugin-based architecture** that makes it easy to:
- Add new exercises without modifying core logic
- Test exercises in isolation
- Swap out components (e.g., different pose detection models)
- Scale horizontally for multiple users

```
exercise-form-correction/
│
├── 📂 frontend/                    # React + TypeScript + Vite
│   ├── 📂 src/
│   │   ├── 📂 components/          # UI Components
│   │   │   ├── ExerciseDisplay.tsx    # Shows current exercise, reps, feedback
│   │   │   ├── VideoCanvas.tsx        # Video player with skeleton overlay
│   │   │   ├── VideoUpload.tsx        # Drag-and-drop file upload
│   │   │   └── ControlPanel.tsx       # Play/pause/reset controls
│   │   │
│   │   ├── 📂 hooks/               # Custom React Hooks
│   │   │   ├── usePoseStream.ts       # WebSocket connection management
│   │   │   └── useVideoProcessor.ts   # Video + pose detection pipeline
│   │   │
│   │   ├── 📂 pose/                # Pose Detection Module
│   │   │   ├── PoseDetector.ts        # MediaPipe wrapper
│   │   │   ├── MotionBuffer.ts        # Temporal smoothing
│   │   │   └── index.ts               # Public exports
│   │   │
│   │   ├── 📂 video/               # Video Source Module
│   │   │   ├── VideoSource.ts         # Abstract video interface
│   │   │   ├── FileVideoSource.ts     # File-based video
│   │   │   └── CameraSource.ts        # Webcam capture
│   │   │
│   │   ├── App.tsx                 # Main application
│   │   └── main.tsx                # Entry point
│   │
│   ├── package.json
│   └── vite.config.ts
│
├── 📂 backend/                     # FastAPI + Python
│   ├── 📂 api/                     # API Layer
│   │   ├── routes.py                  # WebSocket endpoints
│   │   └── upload.py                  # Video upload endpoints
│   │
│   ├── 📂 exercises/               # Exercise Modules (PLUGGABLE)
│   │   ├── base.py                    # BaseExercise abstract class
│   │   ├── squat.py                   # Squat detection & form checks
│   │   ├── pushup.py                  # Push-up module
│   │   ├── bicep_curl.py              # Bicep curl module
│   │   └── __init__.py                # Module registry
│   │
│   ├── 📂 state_machine/           # Session State Management
│   │   └── manager.py                 # IDLE → SCANNING → ACTIVE
│   │
│   ├── 📂 config/                  # Configuration
│   │   └── settings.py                # Environment settings
│   │
│   ├── main.py                     # FastAPI application
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

---

## 🔌 Modular Design Philosophy

### Adding a New Exercise

The system is designed so that **adding a new exercise requires only creating a single file**:

```python
# backend/exercises/lunge.py

from .base import BaseExercise, ExerciseResult, JointAngles

class LungeModule(BaseExercise):
    """Lunge exercise detection and form correction."""
    
    # Define angle thresholds
    FRONT_KNEE_ANGLE_MIN = 80
    FRONT_KNEE_ANGLE_MAX = 100
    
    @property
    def name(self) -> str:
        return "Lunge"
    
    @property
    def required_joints(self) -> list:
        return [JointName.LEFT_KNEE, JointName.RIGHT_KNEE, ...]
    
    def detect_rep_phase(self, landmarks) -> str:
        # Implement phase detection
        pass
    
    def check_form(self, landmarks) -> ExerciseResult:
        # Implement form validation
        pass
```

Then register it in `exercises/__init__.py`:

```python
from .lunge import LungeModule

EXERCISE_MODULES = {
    "squat": SquatModule,
    "pushup": PushupModule,
    "bicep_curl": BicepCurlModule,
    "lunge": LungeModule,  # ← Just add this line!
}
```

### Key Design Patterns

| Pattern | Implementation | Benefit |
|---------|---------------|---------|
| **Strategy** | Exercise modules | Swap algorithms at runtime |
| **State Machine** | Session manager | Clean state transitions |
| **Observer** | Pose callbacks | Decouple detection from UI |
| **Factory** | Exercise registry | Dynamic module loading |
| **Adapter** | Video sources | Unified interface for file/camera |

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.10+ (for backend)
- **Git**

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/shahmir2004/exercise-form-correction.git
cd exercise-form-correction
```

#### 2. Set Up Backend

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

#### 3. Set Up Frontend

```bash
cd frontend
npm install
```

### Running the Application

#### Start Backend Server

```bash
# From project root, with venv activated
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: `http://localhost:8000`

#### Start Frontend Development Server

```bash
# In a new terminal
cd frontend
npm run dev
```

Frontend will be available at: `http://localhost:3000`

### Usage

1. Open `http://localhost:3000` in your browser
2. Click **"Upload Video"** or drag and drop a workout video
3. Click **"Start Processing"** to begin analysis
4. Watch real-time feedback as the video plays:
   - Skeleton overlay on your body
   - Exercise type detection
   - Rep counter
   - Form corrections

---

## 📊 Supported Exercises

| Exercise | Detection | Rep Counting | Form Checks |
|----------|-----------|--------------|-------------|
| **Squat** | ✅ | ✅ | Knee valgus, depth, back angle |
| **Push-up** | ✅ | ✅ | Elbow flare, hip sag, depth |
| **Bicep Curl (Standing/Seated)** | ✅ | ✅ | Elbow drift, body swing (seat-aware), ROM |
| **Alternate Bicep Curl** | ✅ | ✅ | Alternation checks, resting arm extension, left/right balance |
| *Lunge* | 🔜 Planned | - | - |
| *Deadlift* | 🔜 Planned | - | - |
| *Plank* | 🔜 Planned | - | - |

---

## 🔮 Future Roadmap

### Phase 1: Web Application Enhancement (Q1 2026)
- [ ] **Webcam Support** - Real-time camera analysis
- [ ] **User Accounts** - Save workout history
- [ ] **Workout Sessions** - Multi-exercise routines
- [ ] **Progress Tracking** - Charts and statistics
- [ ] **Video Recording** - Save analyzed sessions

### Phase 2: Full Fitness Platform (Q2 2026)
- [ ] **Calorie Management**
  - Food logging with barcode scanning
  - Macro tracking (protein, carbs, fats)
  - Daily calorie goals based on activity
  - Integration with fitness trackers
  
- [ ] **Workout Plans**
  - Pre-built exercise programs
  - Custom routine builder
  - Rest day scheduling
  - Progressive overload tracking

- [ ] **Social Features**
  - Share workouts with friends
  - Challenges and competitions
  - Leaderboards

### Phase 3: Mobile Application (Q3 2026)
- [ ] **React Native App**
  - iOS and Android support
  - Offline mode for form checking
  - Push notifications for reminders
  
- [ ] **Wearable Integration**
  - Apple Watch / Wear OS companion
  - Heart rate monitoring during exercises
  - Automatic rep detection from motion sensors

### Phase 4: AI Enhancements (Q4 2026)
- [ ] **Personalized Coaching**
  - ML-based form improvement suggestions
  - Injury risk prediction
  - Adaptive difficulty
  
- [ ] **Voice Feedback**
  - Real-time audio cues
  - "Lower your hips" spoken during squats
  
- [ ] **3D Pose Analysis**
  - Depth camera support
  - More accurate joint angles

---

## 🛠️ Tech Stack

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **MediaPipe Tasks Vision** - Client-side pose detection

### Backend
- **FastAPI** - Python web framework
- **WebSockets** - Real-time communication
- **Uvicorn** - ASGI server
- **NumPy** - Numerical computations

### ML/AI
- **MediaPipe PoseLandmarker** - 33-point body pose estimation
- **GPU Acceleration** - WebGL-based inference

---

## 📄 API Reference

### WebSocket Endpoint

```
WS /api/ws/pose/{client_id}
```

**Example (Browser / PWA):**

```js
const clientId = "pwa-1";
const ws = new WebSocket(`wss://exercise-form-backend.onrender.com/api/ws/pose/${clientId}`);

ws.onopen = () => console.log("WS connected");
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  console.log(msg.current_exercise, msg.rep_count, msg.violations);
};

// Send MediaPipe pose landmarks (33 items)
ws.send(JSON.stringify({
  landmarks: [...],
  timestamp: Date.now()
}));
```

**Send (Client → Server):**
```json
{
  "landmarks": [
    {"x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.99},
    // ... 33 landmarks total
  ],
  "timestamp": 1234567890.123
}
```

**Receive (Server → Client):**
```json
{
  "state": "active",
  "current_exercise": "bicep_curl",
  "exercise_display": "Bicep Curl",
  "rep_count": 5,
  "rep_phase": "up",
  "is_rep_valid": true,
  "violations": [],
  "corrections": [],
  "correction_message": "Good form!",
  "joint_colors": {
    "left_elbow": "green",
    "right_elbow": "green"
  },
  "confidence": 0.95
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Collaborators

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

**Muhammad Shahmir Ahmed** - [Portfolio](https://shahmir-ahmed.vercel.app) · [GitHub](https://github.com/shahmir2004)

**Abdullah Azher Chaudry** - [Portfolio](https://abdullahch.vercel.app) · [GitHub](https://github.com/ABDULLAHAZHERCH)

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the amazing pose detection model
- [FastAPI](https://fastapi.tiangolo.com/) for the elegant Python framework
- [React](https://reactjs.org/) for the frontend framework

---

<p align="center">
  Made with ❤️ for fitness enthusiasts everywhere
</p>
