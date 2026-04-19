# STRICT IMPLEMENTATION PROMPT: Next.js PWA -> FastAPI Exercise Backend

Use this prompt exactly as-is with your coding agent.

---

You must implement integration only. No planning essay, no architecture discussion, no optional redesigns.

## Objective
Integrate a Next.js PWA client with an existing FastAPI backend for:
1. Client-side pose detection from video frames.
2. Real-time landmark streaming over WebSocket.
3. Live backend feedback rendering (exercise, reps, form corrections, joint colors).
4. Optional chunked upload flow for server-hosted playback.

## Backend Contract (MUST MATCH EXACTLY)

### WebSocket
- Endpoint: /api/ws/pose/{client_id}

Client -> server message:
{
  "landmarks": [
    { "x": number, "y": number, "z": number, "visibility": number }
  ],
  "timestamp": number
}

Server -> client message:
{
  "state": "idle" | "scanning" | "active",
  "current_exercise": string | null,
  "exercise_display": string,
  "rep_count": number,
  "rep_phase": string,
  "is_rep_valid": boolean,
  "violations": string[],
  "corrections": string[],
  "correction_message": string,
  "joint_colors": Record<string, string>,
  "confidence": number,
  "timestamp": number
}

### Upload APIs
- POST /api/upload/init
- POST /api/upload/chunk/{upload_id}?chunk_index={i}
- GET /api/upload/status/{upload_id}
- POST /api/upload/complete/{upload_id}
- DELETE /api/upload/{upload_id}
- GET /api/upload/files
- Uploaded files are served under /uploads

## Non-Negotiable Constraints
1. Do not rename backend response fields.
2. Keep session identity stable (single client_id per workout session).
3. Open WS only during active processing; close on stop/unmount.
4. Send only valid landmark payloads.
5. Keep changes minimal and scoped to integration.
6. Do not refactor unrelated code.
7. Do not change backend behavior unless required for compatibility.

## Required Deliverables (IMPLEMENT ALL)
1. TypeScript contracts:
- PoseLandmark
- WsPoseRequest
- WsFormCorrectionResponse
- Upload request/response types

2. Reusable runtime modules:
- WebSocket integration module/hook with:
  - connect(clientId)
  - disconnect()
  - sendLandmarks(landmarks, timestamp)
  - reconnect handling
- Upload API module for init/chunk/status/complete/cancel/list

3. Processing loop wiring:
- Frame processing at controlled FPS
- Landmark extraction and send path
- Cleanup of loops/resources on unmount

4. UI integration:
- Video playback
- Feedback panel for state/exercise/reps/violations/corrections/message/confidence
- Skeleton overlay color mapping from joint_colors

5. Environment config:
- NEXT_PUBLIC_API_BASE_URL
- Derived WS base URL (wss/ws from https/http)

6. Optional mode (if implemented in same task):
- Upload-first flow with resume and playback from /uploads URL

## Acceptance Tests (MUST PASS)
1. Local video plays and frames are analyzed.
2. WS connects and streams responses continuously while processing.
3. rep_count and rep_phase update live in UI.
4. violations/corrections/correction_message appear unchanged from backend.
5. joint_colors updates overlay colors.
6. Temporary WS disconnect does not crash app and can recover.
7. Cleanup on unmount leaves no active loop/socket leaks.

## Output Format (STRICT)
Return only these sections:
1. Changed files
2. Exact code changes
3. Test steps executed
4. Test results
5. Remaining limitations

Do not include roadmap suggestions unless explicitly asked.

## Existing Backend/Frontend References
- backend/api/routes.py
- backend/api/upload.py
- backend/state_machine/manager.py
- backend/exercises/classifier.py
- backend/exercises/base.py
- backend/main.py
- frontend/src/App.tsx
- frontend/src/hooks/usePoseStream.ts
- frontend/src/hooks/useVideoProcessor.ts
- frontend/src/hooks/useChunkedUpload.ts
- frontend/src/pose/PoseDetector.ts
- frontend/src/components/VideoPlayer.tsx
- frontend/src/components/ExerciseDisplay.tsx
- frontend/src/components/SkeletonOverlay.tsx
