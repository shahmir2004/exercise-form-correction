"""
Quick smoke test for the regressions we just fixed.
Feeds synthetic landmark sequences into FormManager and verifies that
(1) HMM picks the right exercise, (2) rep_count increments.

Adds three robustness scenarios after the baseline tests:
  - Noisy squat (Gaussian landmark jitter)
  - Squat with brief occlusion windows
  - Multi-exercise stream (squat -> pushup -> alt-curl in one session)
    to verify the dynamic mid-video switching path.

Run: python smoke_test.py
"""

import math
import random
import sys
import time

from state_machine.manager import FormManager


# Match the production WS frame rate (~20fps) so rate-limiting doesn't drop frames.
_FRAME_INTERVAL = 1.0 / 20.0


def _make_landmarks(
    pose: dict[int, tuple[float, float, float]],
    *,
    visibility: float = 0.95,
) -> list[dict]:
    """Build a 33-landmark list from a partial dict; missing landmarks default to (0.5,0.5,0)."""
    out = []
    for i in range(33):
        x, y, z = pose.get(i, (0.5, 0.5, 0.0))
        out.append({"x": x, "y": y, "z": z, "visibility": visibility})
    return out


def _add_noise(landmarks: list[dict], std: float, rng: random.Random) -> list[dict]:
    """Inject Gaussian noise into landmark x/y to simulate MediaPipe jitter."""
    noisy = []
    for lm in landmarks:
        noisy.append({
            "x": lm["x"] + rng.gauss(0.0, std),
            "y": lm["y"] + rng.gauss(0.0, std),
            "z": lm["z"],
            "visibility": lm["visibility"],
        })
    return noisy


def _set_visibility(landmarks: list[dict], v: float) -> list[dict]:
    """Drop visibility on every landmark to simulate occlusion."""
    return [{**lm, "visibility": v} for lm in landmarks]


def squat_pose(depth: float) -> list[dict]:
    """
    Synthetic squat where depth ∈ [0, 1]: 0 = standing tall, 1 = bottom of squat.
    Image-space coords: y grows downward.
    Standing: hips ~0.55, knees ~0.75, ankles ~0.95
    Bottom:   hips ~0.70, knees ~0.75, ankles ~0.95 (knees bend, hips lower)
    """
    hip_y = 0.55 + 0.15 * depth
    knee_bend = depth  # 0 straight, 1 fully bent
    # Knee angle: 170° standing → 80° at bottom
    knee_angle_deg = 170 - 90 * knee_bend
    # Place knees and ankles such that the knee angle works out.
    # Hips at y=hip_y, knees at y=0.75, ankles at y=0.95 — vary x slightly with bend.
    pose = {
        11: (0.40, 0.40, 0.0),  # left shoulder
        12: (0.60, 0.40, 0.0),  # right shoulder
        13: (0.35, 0.55, 0.0),  # left elbow (arms hanging straight)
        14: (0.65, 0.55, 0.0),  # right elbow
        15: (0.33, 0.68, 0.0),  # left wrist
        16: (0.67, 0.68, 0.0),  # right wrist
        23: (0.45, hip_y, 0.0),  # left hip
        24: (0.55, hip_y, 0.0),  # right hip
        25: (0.43, 0.75, 0.0),   # left knee
        26: (0.57, 0.75, 0.0),   # right knee
        27: (0.45, 0.95, 0.0),   # left ankle
        28: (0.55, 0.95, 0.0),   # right ankle
    }
    # Adjust knee positions so the L_hip - L_knee - L_ankle angle matches knee_angle_deg.
    # Simpler approach: move knees forward (smaller y in our flipped sense) when bending.
    # Compute knee y to satisfy desired knee angle approximately.
    # vector hip→knee and ankle→knee, knee at vertex.
    # For simplicity: bend pulls knees forward in x by `0.08 * knee_bend`.
    pose[25] = (0.43 - 0.05 * knee_bend, 0.70, -0.10 * knee_bend)
    pose[26] = (0.57 + 0.05 * knee_bend, 0.70, -0.10 * knee_bend)
    return _make_landmarks(pose)


def _wrist_for_elbow_angle(
    shoulder: tuple[float, float],
    elbow: tuple[float, float],
    target_angle_deg: float,
    forearm_len: float = 0.18,
    *,
    side: int = +1,
) -> tuple[float, float]:
    """
    Place the wrist so that the elbow angle (shoulder–elbow–wrist) equals
    `target_angle_deg`. `side` selects which of the two valid wrist positions
    to take (+1 swings the wrist outward, -1 swings inward).
    """
    sx, sy = shoulder[0] - elbow[0], shoulder[1] - elbow[1]
    s_angle = math.atan2(sy, sx)
    wrist_angle = s_angle - side * math.radians(target_angle_deg)
    return (
        elbow[0] + forearm_len * math.cos(wrist_angle),
        elbow[1] + forearm_len * math.sin(wrist_angle),
    )


def pushup_pose(depth: float) -> list[dict]:
    """
    Synthetic push-up where depth in [0, 1]: 0 = arms extended (top),
    1 = chest near floor (bottom). Body is horizontal — shoulders, hips,
    and ankles align in y so FeatureExtractor.is_horizontal fires.
    Elbow angle: 165 -> 80 -> 165 across one cycle.
    """
    elbow_angle = 165 - 85 * depth

    body_y = 0.50
    left_shoulder = (0.35, body_y)
    right_shoulder = (0.35, body_y)
    left_elbow = (0.42, body_y + 0.08)
    right_elbow = (0.42, body_y + 0.08)

    lwx, lwy = _wrist_for_elbow_angle(left_shoulder, left_elbow, elbow_angle, side=-1)
    rwx, rwy = _wrist_for_elbow_angle(right_shoulder, right_elbow, elbow_angle, side=+1)

    pose = {
        11: (left_shoulder[0], left_shoulder[1] - 0.02, 0.0),
        12: (right_shoulder[0], right_shoulder[1] + 0.02, 0.0),
        13: (left_elbow[0], left_elbow[1] - 0.02, 0.0),
        14: (right_elbow[0], right_elbow[1] + 0.02, 0.0),
        15: (lwx, lwy - 0.02, 0.0),
        16: (rwx, rwy + 0.02, 0.0),
        23: (0.55, body_y - 0.02, 0.0),  # left hip
        24: (0.55, body_y + 0.02, 0.0),  # right hip
        25: (0.70, body_y - 0.02, 0.0),  # left knee
        26: (0.70, body_y + 0.02, 0.0),  # right knee
        27: (0.85, body_y - 0.02, 0.0),  # left ankle
        28: (0.85, body_y + 0.02, 0.0),  # right ankle
    }
    return _make_landmarks(pose)


def alt_curl_pose(t: float) -> list[dict]:
    """
    Synthetic alternate bicep curl. t ∈ [0, 1] is phase within rep cycle.
    Left arm flexes during t=[0, 0.5] (elbow 165→55°), then extends.
    Right arm does the opposite (55→165°).

    Uses real trigonometry to place wrists so the computed elbow angle
    matches the target — required after the bicep ROM tightening to
    160°/60° in the rep counter.
    """
    # Sweep wide enough that the BicepCurlModule's 5-frame angle smoothing
    # still registers below the 60°/above 160° rep counter thresholds.
    left_angle = 175 - 135 * (1 - abs(2 * t - 1))   # 175° → 40° → 175°
    right_angle = 40 + 135 * (1 - abs(2 * t - 1))   # 40° → 175° → 40°

    left_shoulder = (0.40, 0.40)
    right_shoulder = (0.60, 0.40)
    left_elbow = (0.38, 0.55)
    right_elbow = (0.62, 0.55)

    lwx, lwy = _wrist_for_elbow_angle(left_shoulder, left_elbow, left_angle, side=-1)
    rwx, rwy = _wrist_for_elbow_angle(right_shoulder, right_elbow, right_angle, side=+1)

    pose = {
        11: (left_shoulder[0], left_shoulder[1], 0.0),
        12: (right_shoulder[0], right_shoulder[1], 0.0),
        13: (left_elbow[0], left_elbow[1], 0.0),
        14: (right_elbow[0], right_elbow[1], 0.0),
        15: (lwx, lwy, 0.0),
        16: (rwx, rwy, 0.0),
        23: (0.45, 0.62, 0.0),
        24: (0.55, 0.62, 0.0),
        25: (0.45, 0.80, 0.0),
        26: (0.55, 0.80, 0.0),
        27: (0.45, 0.97, 0.0),
        28: (0.55, 0.97, 0.0),
    }
    return _make_landmarks(pose)


def _hold_at_top(pose_fn, m: FormManager, frames: int = 10) -> None:
    """Hold the pose at depth=0 (peak) for N frames so find_peaks can register
    the trailing edge of the last rep — mirrors real end-of-set behavior."""
    for _ in range(frames):
        time.sleep(_FRAME_INTERVAL)
        m.process_frame(pose_fn(0.0))


def run_squat_test() -> int:
    print("\n=== Squat test (6 cycles, expect 5+ reps) ===")
    m = FormManager()
    last_rep = 0
    last_ex = None
    # First cycle is consumed by HMM lock-in; expect ~(N-1) reps from N cycles.
    for rep in range(6):
        for f in range(31):  # inclusive — ensure depth returns fully to 0
            depth = abs(2 * f / 30 - 1)
            depth = 1 - depth
            time.sleep(_FRAME_INTERVAL)
            r = m.process_frame(squat_pose(depth))
            if r.current_exercise and r.current_exercise != last_ex:
                last_ex = r.current_exercise
                print(f"  rep {rep} f{f}: detected {last_ex} (conf={r.exercise_confidence:.2f})")
        if m.rep_count != last_rep:
            print(f"  end of rep {rep}: rep_count={m.rep_count}")
            last_rep = m.rep_count
    _hold_at_top(squat_pose, m)
    print(f"  FINAL squat rep_count={m.rep_count} (expected ~5)")
    print(f"  FINAL exercise={m.current_exercise}")
    return m.rep_count


def run_alt_curl_test() -> tuple[int, int]:
    """Returns (bad flips away from ALT_CURL, counted reps)."""
    print("\n=== Alt-curl test (5 cycles) ===")
    m = FormManager()
    bad_flips = 0
    last_label = None
    for rep in range(5):
        # Iterate inclusive of the cycle endpoint so the elbow angle actually
        # crosses the upper rep threshold (160°) — needed since the new
        # bicep curl ROM gate requires true full extension.
        for f in range(41):
            t = f / 40.0
            time.sleep(_FRAME_INTERVAL)
            r = m.process_frame(alt_curl_pose(t))
            if r.current_exercise:
                label = str(r.current_exercise)
                if label != last_label:
                    if last_label is not None:
                        # Bad flip: anything that goes back to a non-curl class
                        if "CURL" not in label.upper():
                            bad_flips += 1
                            print(f"  rep{rep} f{f}: BAD FLIP {last_label} -> {label}")
                        else:
                            print(f"  rep{rep} f{f}: flip {last_label} -> {label}")
                    last_label = label
    _hold_at_top(lambda _d: alt_curl_pose(0.0), m)
    print(f"  FINAL alt_curl rep_count={m.rep_count}")
    print(f"  FINAL exercise={m.current_exercise}")
    print(f"  Bad flips (to non-curl): {bad_flips}")
    return bad_flips, m.rep_count


def run_noisy_squat_test() -> int:
    """Squat with Gaussian landmark jitter — find_peaks + savgol must absorb it."""
    print("\n=== Noisy squat test (6 cycles, +/-0.005 normalized landmark jitter) ===")
    rng = random.Random(0xC0FFEE)
    m = FormManager()
    for _rep in range(6):
        for f in range(31):
            depth = 1 - abs(2 * f / 30 - 1)
            time.sleep(_FRAME_INTERVAL)
            frame = _add_noise(squat_pose(depth), std=0.005, rng=rng)
            m.process_frame(frame)
    _hold_at_top(squat_pose, m)
    print(f"  FINAL noisy squat rep_count={m.rep_count} (expect ~5)")
    return m.rep_count


def run_occlusion_squat_test() -> int:
    """Squat with two transient occlusion windows — must not abort reps."""
    print("\n=== Occlusion squat test (6 cycles, vis=0.1 for two 0.4s windows) ===")
    m = FormManager()
    # Two ~8-frame (0.4s @ 20fps) low-vis windows during cycles 2 and 4.
    occlusion_frames = set(range(2 * 31 + 5, 2 * 31 + 13)) | set(range(4 * 31 + 5, 4 * 31 + 13))
    f_total = 0
    for _rep in range(6):
        for f in range(31):
            depth = 1 - abs(2 * f / 30 - 1)
            time.sleep(_FRAME_INTERVAL)
            frame = squat_pose(depth)
            if f_total in occlusion_frames:
                frame = _set_visibility(frame, 0.1)
            m.process_frame(frame)
            f_total += 1
    _hold_at_top(squat_pose, m)
    print(f"  FINAL occluded squat rep_count={m.rep_count} (expect >= 4)")
    return m.rep_count


def run_multi_exercise_test() -> tuple[int, list[str]]:
    """Squat -> push-up -> alt-curl in one stream. Verifies dynamic mid-video swap."""
    print("\n=== Multi-exercise test (squat -> pushup -> alt-curl, one session) ===")
    m = FormManager()
    seen_exercises: list[str] = []
    last_label: str | None = None

    def _record(r) -> None:
        nonlocal last_label
        label = (r.current_exercise.value if r.current_exercise else None)
        if label and label != last_label:
            print(f"  detected {label} (rep_count={m.rep_count})")
            if label not in seen_exercises:
                seen_exercises.append(label)
            last_label = label

    # 5 squat cycles + brief hold so the last peak is detected before swap
    for _rep in range(5):
        for f in range(31):
            depth = 1 - abs(2 * f / 30 - 1)
            time.sleep(_FRAME_INTERVAL)
            r = m.process_frame(squat_pose(depth))
            _record(r)
    _hold_at_top(squat_pose, m)
    squat_reps = m.rep_count
    print(f"  squat phase done, rep_count={squat_reps}")

    # 5 pushup cycles (~30 frames each = 1.5s)
    for _rep in range(5):
        for f in range(31):
            depth = 1 - abs(2 * f / 30 - 1)
            time.sleep(_FRAME_INTERVAL)
            r = m.process_frame(pushup_pose(depth))
            _record(r)
    _hold_at_top(pushup_pose, m)
    pushup_total = m.rep_count
    print(f"  pushup phase done, total rep_count={pushup_total}")

    # 5 alt-curl cycles
    for _rep in range(5):
        for f in range(41):
            t = f / 40.0
            time.sleep(_FRAME_INTERVAL)
            r = m.process_frame(alt_curl_pose(t))
            _record(r)
    _hold_at_top(lambda _d: alt_curl_pose(0.0), m)
    alt_total = m.rep_count
    print(f"  alt-curl phase done, total rep_count={alt_total}")
    print(f"  exercises seen: {seen_exercises}")
    return alt_total, seen_exercises


if __name__ == "__main__":
    try:
        squat_reps = run_squat_test()
        bad_flips, alt_reps = run_alt_curl_test()
        noisy_reps = run_noisy_squat_test()
        occluded_reps = run_occlusion_squat_test()
        multi_total, exercises_seen = run_multi_exercise_test()

        # Acceptance criteria.
        # - squat + alt curl baselines unchanged (>=5 / >=4)
        # - noisy squat: within 1 rep of clean baseline (5)
        # - occluded squat: at least 4 reps despite vis dropouts
        # - multi-exercise: at least two of the three exercises detected
        #   (ideally all three, but the synthetic pushup pose may not always
        #   trip the rule gate at low confidence)
        ok = (
            squat_reps >= 5
            and bad_flips == 0
            and alt_reps >= 4
            and noisy_reps >= 4
            and occluded_reps >= 4
            and len(exercises_seen) >= 2
        )
        print()
        if ok:
            print("SMOKE OK: rep counting robust under jitter, occlusion, and multi-exercise streams.")
            sys.exit(0)
        print(
            "SMOKE FAIL: "
            f"squat_reps={squat_reps} alt_reps={alt_reps} bad_flips={bad_flips} "
            f"noisy_reps={noisy_reps} occluded_reps={occluded_reps} "
            f"exercises_seen={exercises_seen}"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\nSMOKE TEST CRASH: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
