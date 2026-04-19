# Frontend System Overview

This file explains how the frontend modules work together:
- hooks
- components
- pose module
- video module
- App.tsx orchestration
- backend calls (WebSocket + REST)

## 1) High-level runtime flow

1. User selects a video in `VideoPlayer`.
2. `App.tsx` calls `useVideoProcessor.loadFile(file)`.
3. `useVideoProcessor` creates a `FileVideoSource`, initializes an `HTMLVideoElement`, and prepares pose processing.
4. On play, `useVideoProcessor` starts a frame loop and runs `PoseDetector.processFrame(...)` at target FPS.
5. Pose results update `currentLandmarks` in `App.tsx`.
6. While processing is active, `App.tsx` connects `usePoseStream` to backend WebSocket.
7. `App.tsx` sends landmarks to backend through `sendLandmarks(...)`.
8. Backend replies with form correction payload per frame.
9. `App.tsx` stores response in `formResponse`.
10. UI renders:
- `VideoPlayer` + `SkeletonOverlay` using landmarks and `joint_colors`.
- `ExerciseDisplay` using backend response fields (state, exercise, reps, violations, corrections, confidence).

## 2) App.tsx as orchestrator

Source: `App.tsx`

### Core state

- `currentLandmarks: PoseLandmark[] | null`
- `formResponse: FormCorrectionResponse | null`

### Hooks used

- `useVideoProcessor(...)`:
  - video state and controls
  - pose detector readiness
  - processing status
  - per-frame pose callback

- `usePoseStream(...)`:
  - WebSocket connectivity state
  - send landmarks
  - receive backend correction messages

### Important App.tsx effects

1. WebSocket lifecycle:
- If `isProcessing` becomes true and socket is not connected, call `connect()`.
- If `isProcessing` becomes false and socket is connected, call `disconnect()`.

2. Landmark streaming:
- If `currentLandmarks` exists and socket is connected and processing is active, call:
  - `sendLandmarks(currentLandmarks, performance.now())`

### Rendering split

- Left/main: `VideoPlayer`
  - receives `videoElement`, timeline state, controls
  - receives `landmarks` and backend `joint_colors`

- Right/sidebar: `ExerciseDisplay`
  - receives `formResponse` and `isConnected`
  - displays state/exercise/reps/form quality/corrections/confidence

## 3) Hooks

## useVideoProcessor

Source: `hooks/useVideoProcessor.ts`

Purpose:
- Manages video source lifecycle (load/play/pause/seek).
- Initializes pose detector once on mount.
- Runs frame processing loop with `requestAnimationFrame` and FPS throttling.
- Emits pose results via `onPoseResult` callback.
- Stores recent landmark frames in `MotionBuffer`.

Key return values:
- Video: `videoElement`, `isVideoReady`, `isPlaying`, `duration`, `currentTime`
- Controls: `loadFile`, `loadUrl`, `play`, `pause`, `seek`
- Pose: `isPoseReady`, `isProcessing`, `startProcessing`, `stopProcessing`
- Data: `motionBuffer`, `error`

Behavior details:
- Processing starts automatically when video is playing and detector is ready.
- Stops when paused/ended/unmounted.
- Clears motion buffer when loading new source.

## usePoseStream

Source: `hooks/usePoseStream.ts`

Purpose:
- Manages backend WebSocket session for pose analysis.
- Sends outbound landmarks and receives correction responses.
- Tracks connection state and latest response.

Connection model:
- URL: `WS_URL + '/' + clientId`
- `clientId` is stable for hook lifetime (provided or generated once).
- Prevents duplicate parallel connects.

Queue model:
- If socket is not open, outbound messages are queued.
- Queue is capped (last 5 messages kept).
- Queue is flushed on successful `onopen`.

Additional API:
- `reset()` sends `POST /api/reset/{client_id}` to reset backend session state.

## useChunkedUpload

Source: `hooks/useChunkedUpload.ts`

Purpose:
- Uploads videos in chunks with resume/pause/cancel/retry.

Backend calls used:
- `POST /api/upload/init`
- `GET /api/upload/status/{upload_id}`
- `POST /api/upload/chunk/{upload_id}?chunk_index=i`
- `POST /api/upload/complete/{upload_id}`
- `DELETE /api/upload/{upload_id}`

Retry + pause behavior:
- Retries failed chunk uploads with exponential backoff.
- Uses `AbortController` to pause/cancel active requests.
- Resume re-runs flow and skips already uploaded chunks.

Note:
- `VideoUpload` component uses this hook, but `VideoUpload` is not mounted in current `App.tsx`.

## 4) Components

## VideoPlayer

Source: `components/VideoPlayer.tsx`

Responsibilities:
- File drop/select UI for video input.
- Attach `videoElement` created by hook into a DOM container.
- Show playback controls (play/pause/seek/restart/load new).
- Render `SkeletonOverlay` on top of the video.

Inputs from App:
- `videoElement`, `isPlaying`, `currentTime`, `duration`
- `landmarks`
- `jointColors`
- callbacks: `onPlay`, `onPause`, `onSeek`, `onFileSelect`

## SkeletonOverlay

Source: `components/SkeletonOverlay.tsx`

Responsibilities:
- Draw pose connections and joints on a canvas.
- Color joints/connections based on backend `joint_colors`.

Color behavior:
- Converts joint names from landmark index using `getLandmarkName(index)`.
- Uses backend values like `green/red/yellow` to color points and edges.
- Skips low-visibility landmarks and most face points for clarity.

## ExerciseDisplay

Source: `components/ExerciseDisplay.tsx`

Responsibilities:
- Display backend exercise state and feedback panel.

UI states:
- Not connected: “Connecting to server...”
- Connected, no response: “Waiting for video...”
- Active response:
  - `exercise_display`
  - `rep_count`
  - form status from `is_rep_valid` + `violations`
  - `correction_message`
  - list of `violations`
  - `rep_phase`
  - `confidence`

## VideoUpload

Source: `components/VideoUpload.tsx`

Responsibilities:
- UX wrapper over `useChunkedUpload`.
- Shows chunk upload progress and controls.

Integration status:
- Implemented, but not currently used in `App.tsx`.

## 5) Pose module

## PoseDetector

Source: `pose/PoseDetector.ts`

Responsibilities:
- Wraps `@mediapipe/tasks-vision` PoseLandmarker.
- Lazy async initialization (`initialize()`).
- Per-frame inference (`detectForVideo`) and callback emission.

Output:
- `PoseResult`:
  - `landmarks` (normalized 2D + z + visibility)
  - `worldLandmarks`
  - `timestamp`

Guardrails:
- Skips processing if not initialized, frame is not ready, or previous frame is still processing.

## MotionBuffer

Source: `pose/MotionBuffer.ts`

Responsibilities:
- Maintains rolling frame history.
- Provides local helper metrics:
  - displacement
  - velocity
  - significant motion detection
  - buffer duration

Current use:
- Populated by `useVideoProcessor` on every pose result.

## 6) Video module

## VideoSource (abstract)

Source: `video/VideoSource.ts`

Defines shared interface for all video sources:
- `initialize`, `play`, `pause`, `stop`, `seek`
- metadata methods and source capability flags

## FileVideoSource

Source: `video/FileVideoSource.ts`

Responsibilities:
- Wrap local file or URL as video source.
- Produces an `HTMLVideoElement` for player.
- Supports seek and metadata.

Current app usage:
- `useVideoProcessor` uses `FileVideoSource.fromFile(...)` for uploaded files.

## WebcamSource

Source: `video/WebcamSource.ts`

Responsibilities:
- Access camera via `getUserMedia`.
- Expose live video stream as source.

Current app usage:
- Available in codebase, not currently wired in `App.tsx`.

## VideoSourceFactory

Source: `video/VideoSourceFactory.ts`

Responsibilities:
- Helper to construct source type (`file` or `webcam`).

Current app usage:
- `useVideoProcessor` currently instantiates `FileVideoSource` directly.

## 7) Backend calls from frontend

## WebSocket (real-time analysis)

Where:
- Constructed in `usePoseStream`
- Triggered by `App.tsx` during active processing

Endpoint pattern:
- `${WS_URL}/${clientId}`
- `WS_URL` derives from frontend config (ws/wss + backend base path)

Outbound frame payload:
{
  "landmarks": [
    { "x": number, "y": number, "z": number, "visibility": number }
  ],
  "timestamp": number
}

Inbound analysis payload (stored in `formResponse`):
- `state`
- `current_exercise`
- `exercise_display`
- `rep_count`
- `rep_phase`
- `is_rep_valid`
- `violations`
- `corrections`
- `correction_message`
- `joint_colors`
- `confidence`
- `timestamp`

How frontend uses response:
- `ExerciseDisplay` consumes text/status/rep/quality fields.
- `VideoPlayer -> SkeletonOverlay` consumes `joint_colors` to color joints/connections.

## REST endpoints

1. Session reset (from `usePoseStream.reset`):
- `POST /api/reset/{client_id}`

2. Upload flow (from `useChunkedUpload`):
- `POST /api/upload/init`
- `GET /api/upload/status/{upload_id}`
- `POST /api/upload/chunk/{upload_id}?chunk_index=i`
- `POST /api/upload/complete/{upload_id}`
- `DELETE /api/upload/{upload_id}`

## 8) End-to-end data path (concise)

1. Video frame -> `PoseDetector` -> landmarks.
2. Landmarks -> `App.tsx` state (`currentLandmarks`).
3. `App.tsx` -> `usePoseStream.sendLandmarks(...)`.
4. Backend analyzes and responds.
5. Response -> `App.tsx` state (`formResponse`).
6. `formResponse` -> `ExerciseDisplay` + `SkeletonOverlay` rendering.

## 9) Important implementation notes

- WebSocket is intentionally not auto-connected; `App.tsx` ties connection to processing state.
- `performance.now()` is used as frame timestamp in outbound WS messages.
- Current user-facing flow is local file analysis with real-time backend feedback.
- Upload-to-backend flow exists separately and can be integrated into `App.tsx` if needed.
