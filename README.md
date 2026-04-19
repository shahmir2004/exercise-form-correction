# Backend

FastAPI backend for real-time exercise form correction.

This service receives pose landmarks from the frontend over WebSocket, classifies the exercise, runs the active exercise module, and returns rep state, form feedback, and joint highlighting data. It also exposes chunked upload endpoints for video analysis.

## Overview

The backend is responsible for:

- Accepting pose landmark frames from clients.
- Managing per-client exercise state.
- Classifying the active exercise.
- Running exercise-specific form checks and rep counting.
- Serving upload APIs for chunked video ingestion.
- Exposing a health endpoint for deployment checks.

## Tech Stack

- FastAPI for HTTP and WebSocket routing.
- Uvicorn as the ASGI server.
- Pydantic for request and response validation.
- aiofiles for async file handling.
- NumPy for pose and angle math.

## Project Structure

```text
backend/
├── main.py               # FastAPI application entry point
├── api/
│   ├── routes.py         # WebSocket pose endpoint + health/reset routes
│   └── upload.py         # Chunked upload lifecycle
├── config/
│   └── settings.py       # Environment-driven settings
├── database/
│   └── client.py         # Database abstraction
├── exercises/
│   ├── base.py           # Base exercise contract
│   ├── classifier.py    # Exercise identification
│   ├── squat.py         # Squat module
│   ├── pushup.py        # Push-up module
│   └── bicep_curl.py    # Bicep curl modules
├── state_machine/
│   └── manager.py        # Session and exercise state transitions
├── utils/
│   ├── rep_counter.py    # Hysteresis-based rep counting
│   └── smoothing.py      # Landmark and angle smoothing
└── uploads/              # Stored uploaded videos and chunk data
```

## Local Setup

### Prerequisites

- Python 3.10 or newer.
- A virtual environment is recommended.

### Install Dependencies

From the repository root:

```bash
pip install -r backend/requirements.txt
```

### Run the API

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Verify It Is Running

- Root: `GET /`
- Health: `GET /api/health`
- OpenAPI docs: `GET /docs`

## Environment Variables

The backend reads settings from environment variables and a local `.env` file.

### Common Settings

- `HOST`: bind address for local development. Default: `0.0.0.0`
- `PORT`: server port. Default: `8000`
- `DEBUG`: reload/debug mode. Default: `True`
- `CORS_ORIGINS`: comma-separated allowed frontend origins or `*`
- `UPLOAD_DIR`: directory for completed uploads. Default: `./uploads`
- `CHUNK_DIR`: directory for temporary chunk storage. Default: `./uploads/chunks`
- `MAX_FILE_SIZE`: maximum upload size. Default: `5GB`
- `CHUNK_SIZE`: upload chunk size. Default: `5MB`
- `MOTION_BUFFER_SIZE`: frame buffer size used by motion analysis. Default: `60`
- `CONFIDENCE_THRESHOLD`: minimum confidence for exercise detection. Default: `0.80`
- `EXERCISE_SWITCH_DELAY`: delay before switching exercises. Default: `2.0`

## CORS

The backend allows local development origins and Vercel-style origins by default. For production, set `CORS_ORIGINS` explicitly to your deployed frontend URL(s).

Example:

```bash
CORS_ORIGINS=https://your-frontend.vercel.app,https://www.yourdomain.com
```

## WebSocket API

### Endpoint

```text
WS /api/ws/pose/{client_id}
```

`client_id` should be a stable unique identifier per browser session or device.

### Client To Server Payload

The client sends a JSON object with pose landmarks and a timestamp:

```json
{
  "landmarks": [
    {"x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.99}
  ],
  "timestamp": 1234567890.123
}
```

The landmarks list should contain 33 MediaPipe pose landmarks.

### Server Response

The backend returns a `FormCorrectionResponse` payload containing:

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

Example:

```json
{
  "state": "active",
  "current_exercise": "squat",
  "exercise_display": "Squat",
  "rep_count": 3,
  "rep_phase": "down",
  "is_rep_valid": true,
  "violations": [],
  "corrections": [],
  "correction_message": "Great form! Keep it up!",
  "joint_colors": {},
  "confidence": 0.94,
  "timestamp": 1234567890.123
}
```

## HTTP API

### Root

`GET /`

Returns the service name, version, docs path, and health path.

### Health Check

`GET /api/health`

Returns a simple healthy response and active WebSocket connection count.

### Reset Session

`POST /api/reset/{client_id}`

Resets the session state for a specific client.

## Upload API

The upload flow supports chunked file upload for larger workout videos.

### Initialize Upload

`POST /api/upload/init`

Creates a new upload session and returns an `upload_id` plus server chunk size.

### Upload A Chunk

`POST /api/upload/chunk/{upload_id}?chunk_index=N`

Uploads one file chunk for the session.

### Check Status

`GET /api/upload/status/{upload_id}`

Returns progress and uploaded chunk indices.

### Complete Upload

`POST /api/upload/complete/{upload_id}`

Assembles all chunks into the final file and schedules cleanup.

### Cancel Upload

`DELETE /api/upload/{upload_id}`

Cancels the upload session and removes stored chunks.

## Deployment Notes

This backend is an always-on API and WebSocket server. It is not a good fit for Vercel serverless deployment because the application keeps long-lived WebSocket connections open.

Recommended deployment targets:

- Render
- Fly.io
- Railway
- A small VPS or container host

If you deploy the frontend on Vercel, point the browser to the backend WebSocket endpoint using the backend deployment URL, for example:

```text
wss://your-backend-domain/api/ws/pose/{client_id}
```

## Frontend Integration

Set the frontend API base URL to the backend deployment:

```bash
VITE_API_URL=https://your-backend-domain
```

The frontend derives the WebSocket URL from this value.

## Troubleshooting

- If the WebSocket does not connect, confirm `VITE_API_URL` points to the backend and not the frontend.
- If the browser blocks the connection, verify `CORS_ORIGINS` includes the deployed frontend domain.
- If uploads fail, confirm the backend process has write access to `UPLOAD_DIR` and `CHUNK_DIR`.
- If rep counting or exercise switching seems unstable, check the smoothing and state machine thresholds in `config/settings.py` and the exercise modules.

## Related Files

- [backend/main.py](main.py)
- [backend/api/routes.py](api/routes.py)
- [backend/api/upload.py](api/upload.py)
- [backend/config/settings.py](config/settings.py)
- [backend/exercises/classifier.py](exercises/classifier.py)
- [backend/state_machine/manager.py](state_machine/manager.py)
