# Next.js PWA Integration Guide + Agent Prompt

## 1) Technical Workflow Overview (Current System)

This project has two pipelines:

1. Real-time analysis pipeline (currently wired in frontend app).
2. Chunked upload pipeline (implemented, but not currently mounted in main app flow).

### A. Real-time analysis pipeline

1. User selects a video file in the UI.
2. Video is loaded into an HTMLVideoElement.
3. MediaPipe Pose Landmarker runs on the client at target FPS (default 30).
4. For each processed frame, client sends landmarks to backend via WebSocket.
5. Backend keeps a per-client FormManager session state.
6. State machine transitions: idle -> scanning -> active.
7. Classifier analyzes motion buffer and identifies exercise type.
8. Active exercise module computes rep phase, rep count, violations, corrections, joint colors, confidence.
9. Backend sends FormCorrectionResponse on every frame.
10. Frontend renders response in status panel and skeleton overlay.

### B. Chunked upload pipeline

1. Initialize upload session.
2. Upload chunks by index.
3. Resume with status endpoint if interrupted.
4. Complete upload to assemble final file.
5. Backend serves uploaded files under /uploads.

Note:
- Current main app uses local file playback + real-time WS analysis.
- Upload component exists but is not currently mounted in main app.

## 2) Backend Contracts

### WebSocket endpoint

- Path: /api/ws/pose/{client_id}
- Client sends JSON:

{
  "landmarks": [
    { "x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.99 }
  ],
  "timestamp": 123456.78
}

Expected:
- landmarks should be MediaPipe pose landmarks (33 points).

Server returns JSON per frame:

{
  "state": "idle | scanning | active",
  "current_exercise": "squat | pushup | bicep_curl | alternate_bicep_curl | null",
  "exercise_display": "string",
  "rep_count": 0,
  "rep_phase": "idle | down | up | hold | transition | unknown",
  "is_rep_valid": true,
  "violations": ["string"],
  "corrections": ["string"],
  "correction_message": "string",
  "joint_colors": { "left_elbow": "green", "right_knee": "red" },
  "confidence": 0.0,
  "timestamp": 123456.78
}

### Upload endpoints

- POST /api/upload/init
  body: { filename, total_size, total_chunks, file_hash? }
  returns: { upload_id, chunk_size }

- POST /api/upload/chunk/{upload_id}?chunk_index={i}
  multipart form-data with chunk
  returns: chunk progress status

- GET /api/upload/status/{upload_id}
  returns uploaded chunk indices and progress

- POST /api/upload/complete/{upload_id}
  assembles chunks and returns final file metadata

- DELETE /api/upload/{upload_id}
  cancels and cleans session

- GET /api/upload/files
  lists uploaded videos

## 3) Integration Requirements for Next.js PWA

1. Create a stable client_id per user/session, do not regenerate every frame.
2. Open WS only while analysis is active, close it on stop/unmount.
3. Send landmarks only after detector is ready and frame has valid landmark list.
4. Keep outbound payload schema exactly matching backend expectation.
5. Treat backend response as source of truth for state, rep_count, and correction text.
6. Render joint_colors onto overlay using landmark names compatible with MediaPipe index mapping.
7. If using upload-first flow, convert returned file metadata into browser-playable URL under /uploads.
8. Handle reconnect and transient WS errors without losing session identity.
9. Respect backend CORS and environment base URL configuration.

## 4) Agent Task Prompt (Copy/Paste)

You are integrating a Next.js PWA client with an existing FastAPI exercise form-correction backend.

Goal:
Implement production-ready integration for video playback, client-side pose detection, WebSocket landmark streaming, and live feedback rendering.

Backend behavior to target:
- WebSocket endpoint: /api/ws/pose/{client_id}
- Upload endpoints: /api/upload/init, /api/upload/chunk/{upload_id}, /api/upload/status/{upload_id}, /api/upload/complete/{upload_id}, /api/upload/{upload_id}
- Uploaded files served from /uploads
- Response fields required from WS:
  state, current_exercise, exercise_display, rep_count, rep_phase,
  is_rep_valid, violations, corrections, correction_message,
  joint_colors, confidence, timestamp

Implement these deliverables:
1. A typed integration layer in Next.js:
   - TypeScript interfaces for WS request/response.
   - API utilities for upload flow.
2. A resilient WebSocket hook/module:
   - connect(client_id), disconnect(), sendLandmarks(landmarks, timestamp).
   - reconnect strategy and queue/drop policy for outbound frames.
3. A pose-processing loop:
   - process video frames at controlled FPS.
   - send only valid landmarks payloads.
4. UI wiring:
   - video player, feedback panel, and overlay colored by joint_colors.
5. Optional upload-first mode:
   - chunked upload + resume + completion + playback from /uploads URL.
6. Environment config:
   - NEXT_PUBLIC_API_BASE_URL
   - derived WS URL.
7. Error handling and cleanup:
   - on unmount, stop loops, close sockets, release resources.

Non-negotiable constraints:
- Preserve exact WS payload contract and response typing.
- Do not rename backend fields.
- Keep session/client identity stable during a workout session.
- No broad refactors unrelated to integration.

Acceptance criteria:
1. App can analyze a local video and display live backend feedback.
2. Rep count increments and rep phase changes are reflected in UI.
3. Violations/corrections and correction_message are rendered as-is.
4. Overlay colors react to joint_colors response values.
5. Connection recovers from temporary disconnect without crashing.
6. Optional upload mode can init, upload chunks, complete, and play uploaded video.

Output format required from you:
1. Brief architecture summary.
2. File-by-file change list.
3. Final TypeScript types for contracts.
4. Manual test steps and expected outcomes.
5. Known limitations and next improvements.

## 5) Source Map (for implementation trace)

Frontend orchestration:
- frontend/src/App.tsx
- frontend/src/hooks/useVideoProcessor.ts
- frontend/src/hooks/usePoseStream.ts
- frontend/src/pose/PoseDetector.ts
- frontend/src/components/VideoPlayer.tsx
- frontend/src/components/ExerciseDisplay.tsx
- frontend/src/components/SkeletonOverlay.tsx
- frontend/src/hooks/useChunkedUpload.ts

Backend flow:
- backend/api/routes.py
- backend/api/upload.py
- backend/state_machine/manager.py
- backend/exercises/classifier.py
- backend/exercises/base.py
- backend/exercises/squat.py
- backend/exercises/pushup.py
- backend/exercises/bicep_curl.py
- backend/main.py
- backend/config/settings.py
