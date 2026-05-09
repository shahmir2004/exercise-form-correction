"""
End-to-end WebSocket smoke test against a running backend.

Mimics what gymi's `usePoseWebSocket` hook sends per frame -- a JSON message
with `landmarks` (list of {x, y, z, visibility}) and a millisecond `timestamp`.
Streams synthetic squat -> push-up -> alt-curl landmarks across one connection
and asserts that the server emits sane state, exercise, and rep_count values.

Run:
  python scripts/ws_replay_smoke.py [--url ws://127.0.0.1:8001/api/ws/pose/smoke]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

# Allow importing the synthetic pose builders from smoke_test.
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent))

from smoke_test import (  # noqa: E402
    _hold_at_top,
    alt_curl_pose,
    pushup_pose,
    squat_pose,
)

import websockets  # noqa: E402


_FRAME_HZ = 20.0
_FRAME_DT = 1.0 / _FRAME_HZ


async def _send_frames(ws, pose_fn, cycle_frames: int, n_cycles: int) -> None:
    for _rep in range(n_cycles):
        for f in range(cycle_frames):
            depth = 1 - abs(2 * f / (cycle_frames - 1) - 1)
            payload = {
                "landmarks": pose_fn(depth),
                "timestamp": time.time() * 1000.0,
            }
            await ws.send(json.dumps(payload))
            await asyncio.sleep(_FRAME_DT)


async def _hold_frames(ws, pose_fn, n_frames: int = 12) -> None:
    for _ in range(n_frames):
        payload = {
            "landmarks": pose_fn(0.0),
            "timestamp": time.time() * 1000.0,
        }
        await ws.send(json.dumps(payload))
        await asyncio.sleep(_FRAME_DT)


async def _drain(ws, into: list[dict], stop_event: asyncio.Event) -> None:
    """Consume server messages until stop_event is set."""
    try:
        while not stop_event.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            try:
                into.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass


async def run(url: str) -> int:
    print(f"connecting to {url}")
    async with websockets.connect(url, max_size=4 * 1024 * 1024) as ws:
        responses: list[dict] = []
        stop = asyncio.Event()
        drain_task = asyncio.create_task(_drain(ws, responses, stop))

        # squat -> hold -> pushup -> hold -> alt-curl -> hold
        await _send_frames(ws, squat_pose, cycle_frames=31, n_cycles=5)
        await _hold_frames(ws, squat_pose)
        await _send_frames(ws, pushup_pose, cycle_frames=31, n_cycles=5)
        await _hold_frames(ws, pushup_pose)
        await _send_frames(ws, alt_curl_pose, cycle_frames=41, n_cycles=5)
        await _hold_frames(ws, lambda _d: alt_curl_pose(0.0))

        # Allow trailing responses to arrive.
        await asyncio.sleep(0.5)
        stop.set()
        await drain_task

    if not responses:
        print("FAIL: no server responses")
        return 1

    final = responses[-1]
    print(f"received {len(responses)} server frames")
    print(f"final state: {final.get('state')}")
    print(f"final exercise: {final.get('current_exercise')}")
    print(f"final rep_count: {final.get('rep_count')}")
    print(f"final phase: {final.get('rep_phase')}")
    print(f"final confidence: {final.get('confidence', 0):.2f}")

    exercises_seen: list[str] = []
    rep_max_per_exercise: dict[str, int] = {}
    last_ex: str | None = None
    last_phase: str | None = None
    phase_changes = 0

    for r in responses:
        ex = r.get("current_exercise")
        if ex and ex != last_ex:
            print(f"  -> {ex}  (rep_count={r.get('rep_count')}, conf={r.get('confidence',0):.2f})")
            if ex not in exercises_seen:
                exercises_seen.append(ex)
            last_ex = ex
        if ex:
            rep_max_per_exercise[ex] = max(
                rep_max_per_exercise.get(ex, 0), int(r.get("rep_count") or 0)
            )
        phase = r.get("rep_phase")
        if phase and phase != last_phase:
            phase_changes += 1
            last_phase = phase

    print(f"\nexercises seen: {exercises_seen}")
    print(f"rep_count high-water per exercise: {rep_max_per_exercise}")
    print(f"phase transitions: {phase_changes}")

    ok = (
        len(exercises_seen) >= 2
        and any(v >= 3 for v in rep_max_per_exercise.values())
        and phase_changes >= 4  # rep counter should swing through ECC/CONC/HOLD
    )
    if ok:
        print("\nE2E OK")
        return 0
    print("\nE2E FAIL")
    return 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--url",
        default=f"ws://127.0.0.1:8001/api/ws/pose/replay-{uuid.uuid4().hex[:8]}",
    )
    args = p.parse_args()
    return asyncio.run(run(args.url))


if __name__ == "__main__":
    sys.exit(main())
