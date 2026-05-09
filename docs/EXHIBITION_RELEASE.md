# Exhibition Release Notes

This release stabilizes the live Gymi form-checking backend for local exhibition use.

## Backend behavior

- Squat rule gating now prioritizes lower-body evidence, so bent hands in front of the body do not force a bicep-curl classification.
- Rep counting uses the existing smoothed peak detector plus a threshold fallback for short, clean demo reps.
- Phase display treats low-amplitude landmark jitter as `hold` or `setup`, including mid-range stillness and top/bottom holds.
- Bicep curl analysis supports upper-body-only framing. Lower-body-only checks are skipped when hips/knees are not visible.
- WebSocket requests may include `camera_view` as `auto`, `front`, `side`, or `three_quarter`.
- WebSocket responses include the resolved `camera_view`.

## Local verification

From `form-checking-backend/backend`:

```bash
python -m pytest
```

Expected result for this release:

```text
69 passed
```

Run locally:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

