# Frontend Hooks Reference

This folder contains the reusable React hooks that drive video ingestion, pose processing, backend streaming, and upload workflows.

## Overview

- `useVideoProcessor`: Manages video source lifecycle and MediaPipe frame processing.
- `usePoseStream`: Manages WebSocket connection and landmark streaming to backend.
- `useChunkedUpload`: Manages resumable chunked file upload to backend.
- `index.ts`: Barrel exports for all hooks and related TypeScript types.

## useVideoProcessor

Source: `useVideoProcessor.ts`

Purpose:
- Owns video element creation and playback control.
- Initializes the pose detector once.
- Runs a frame loop at a target FPS and emits pose results.
- Keeps a local motion buffer of recent landmarks.

### Inputs (`UseVideoProcessorOptions`)

- `onPoseResult?: (result: PoseResult) => void`
- `onError?: (error: Error) => void`
- `targetFps?: number` (default `30`)
- `bufferSize?: number` (default `60`)

### Returns (`UseVideoProcessorReturn`)

Video state:
- `videoElement`
- `isVideoReady`
- `isPlaying`
- `duration`
- `currentTime`

Video controls:
- `loadFile(file)`
- `loadUrl(url)`
- `play()`
- `pause()`
- `seek(time)`

Pose state:
- `isPoseReady`
- `isProcessing`
- `startProcessing()`
- `stopProcessing()`

Other:
- `motionBuffer`
- `error`

### Internal behavior

- Creates one `PoseDetector` on mount.
- Registers detector callback and forwards each `PoseResult` to `onPoseResult`.
- Uses `requestAnimationFrame` + FPS throttling to avoid over-processing.
- Automatically starts processing when video is playing and detector is ready.
- Stops processing when paused or on unmount.

### Typical usage

```tsx
const processor = useVideoProcessor({
  onPoseResult: (result) => {
    // result.landmarks -> send to backend or render overlay
  },
  onError: console.error,
  targetFps: 30,
});

await processor.loadFile(file);
await processor.play();
```

## usePoseStream

Source: `usePoseStream.ts`

Purpose:
- Creates and manages a WebSocket session per client.
- Sends landmark frames to backend.
- Receives and stores latest form-correction response.
- Supports manual connect/disconnect and optional auto-connect.

### Inputs (`UsePoseStreamOptions`)

- `url?: string` (defaults to `WS_URL` from config)
- `clientId?: string`
- `autoConnect?: boolean` (default `false`)
- `onResponse?: (response: FormCorrectionResponse) => void`
- `onError?: (error: Error) => void`
- `onConnect?: () => void`
- `onDisconnect?: () => void`

### Returns (`UsePoseStreamReturn`)

- `isConnected`
- `isConnecting`
- `latestResponse`
- `error`
- `connect()`
- `disconnect()`
- `sendLandmarks(landmarks, timestamp)`
- `reset()`

### Internal behavior

- Uses a stable `clientId` (provided or generated once).
- Prevents duplicate parallel connections.
- Queues outgoing messages if socket is not open yet.
- Limits queued messages to the latest 5 entries.
- On successful open, flushes queued messages.
- `reset()` calls backend `POST /api/reset/{client_id}`.

### Message contract used by this hook

Outbound:
- `landmarks`: mapped to `{x, y, z, visibility}`
- `timestamp`: number

Inbound (`FormCorrectionResponse`):
- `state`, `current_exercise`, `exercise_display`
- `rep_count`, `rep_phase`, `is_rep_valid`
- `violations`, `corrections`, `correction_message`
- `joint_colors`, `confidence`, `timestamp`

### Typical usage

```tsx
const stream = usePoseStream({
  autoConnect: false,
  onResponse: setFormResponse,
  onError: console.error,
});

stream.connect();
stream.sendLandmarks(landmarks, performance.now());
```

## useChunkedUpload

Source: `useChunkedUpload.ts`

Purpose:
- Uploads large video files in chunks.
- Supports pause/resume/cancel.
- Retries failed chunk uploads with exponential backoff.

### Inputs (`UseChunkedUploadOptions`)

- `chunkSize?: number` (default `5MB`)
- `maxRetries?: number` (default `3`)
- `onProgress?: (progress: UploadProgress) => void`
- `onComplete?: (result) => void`
- `onError?: (error: Error) => void`

### Returns (`UseChunkedUploadReturn`)

- `upload(file)`
- `pause()`
- `resume(file)`
- `cancel()`
- `progress`
- `isUploading`

### Progress model (`UploadProgress`)

- `progress` (0-100)
- `uploadedChunks`
- `totalChunks`
- `status`: `idle | initializing | uploading | paused | completing | complete | error`
- Optional: `error`, `uploadId`, `filePath`

### Backend flow used by this hook

1. `POST /api/upload/init`
2. `GET /api/upload/status/{upload_id}` (resume check)
3. `POST /api/upload/chunk/{upload_id}?chunk_index=i` for each chunk
4. `POST /api/upload/complete/{upload_id}`
5. Optional cancel: `DELETE /api/upload/{upload_id}`

### Internal behavior

- Uses `AbortController` for pause/cancel.
- On resume, re-runs upload logic and skips already uploaded chunks.
- Retries chunk failures with backoff: `1s`, `2s`, `4s`...

### Typical usage

```tsx
const uploader = useChunkedUpload({
  onComplete: (result) => console.log(result.file_path),
  onError: console.error,
});

await uploader.upload(file);
```

## index.ts (Barrel Exports)

Source: `index.ts`

- Re-exports all hook functions.
- Re-exports all hook option/return/type interfaces.

Use this import style:

```tsx
import { useVideoProcessor, usePoseStream, useChunkedUpload } from '../hooks';
```

## How hooks work together in app flow

1. `useVideoProcessor` emits landmarks from each processed frame.
2. `usePoseStream` sends those landmarks to backend WebSocket and receives feedback.
3. UI renders:
- skeleton overlay from landmarks + `joint_colors`
- exercise/rep/form feedback from latest backend response
4. Optional: `useChunkedUpload` can upload raw video files to backend storage when needed.
