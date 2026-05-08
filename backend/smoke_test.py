"""
Quick smoke test for the regressions we just fixed.
Feeds synthetic landmark sequences into FormManager and verifies that
(1) HMM picks the right exercise, (2) rep_count increments.

Run: python smoke_test.py
"""

import math
import sys
import time

from state_machine.manager import FormManager


# Match the production WS frame rate (~20fps) so rate-limiting doesn't drop frames.
_FRAME_INTERVAL = 1.0 / 20.0


def _make_landmarks(pose: dict[int, tuple[float, float, float]]) -> list[dict]:
    """Build a 33-landmark list from a partial dict; missing landmarks default to (0.5,0.5,0)."""
    out = []
    for i in range(33):
        x, y, z = pose.get(i, (0.5, 0.5, 0.0))
        out.append({"x": x, "y": y, "z": z, "visibility": 0.95})
    return out


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


def run_squat_test() -> None:
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
    print(f"  FINAL alt_curl rep_count={m.rep_count}")
    print(f"  FINAL exercise={m.current_exercise}")
    print(f"  Bad flips (to non-curl): {bad_flips}")
    return bad_flips, m.rep_count


if __name__ == "__main__":
    try:
        squat_reps = run_squat_test()
        bad_flips, alt_reps = run_alt_curl_test()
        ok = (squat_reps >= 5) and (bad_flips == 0) and (alt_reps >= 8)
        print()
        if ok:
            print("SMOKE OK: squat and alt-curl reps counted with stable exercise detection.")
            sys.exit(0)
        print(f"SMOKE FAIL: squat_reps={squat_reps} alt_reps={alt_reps} bad_flips={bad_flips}")
        sys.exit(1)
    except Exception as e:
        print(f"\nSMOKE TEST CRASH: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
