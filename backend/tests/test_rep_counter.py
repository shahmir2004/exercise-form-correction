from exercises.bicep_curl import BicepCurlModule, AlternateBicepCurlModule
from exercises.pushup import PushupModule
from exercises.squat import SquatModule
from pipeline.rep_counter import HysteresisRepCounter


def test_squat_counter_counts_full_cycles_with_eccentric_concentric_phases():
    counter = HysteresisRepCounter(upper_threshold=150, lower_threshold=90, min_rep_duration=0)
    phases = []
    completed = 0

    for _ in range(5):
        for angle in (170, 130, 80, 100, 160):
            phase, done = counter.update(angle, visibility=1.0)
            phases.append(phase.value)
            completed += int(done)

    assert completed == 5
    assert "eccentric" in phases
    assert "concentric" in phases


def test_partial_reps_and_threshold_jitter_do_not_increment():
    counter = HysteresisRepCounter(upper_threshold=150, lower_threshold=90, min_rep_duration=0)

    for angle in (170, 145, 151, 146, 152, 148, 151):
        counter.update(angle, visibility=1.0)

    assert counter.rep_count == 0
    assert counter.partial_reps == 0


def test_low_visibility_aborts_in_flight_rep():
    """Visibility drop mid-rep must reset the counter, not count the rep."""
    counter = HysteresisRepCounter(upper_threshold=150, lower_threshold=90, min_rep_duration=0)
    # Start a rep with full visibility
    for angle in (170, 130, 80):
        counter.update(angle, visibility=1.0)
    # Visibility drops for 3+ frames mid-rep
    for _ in range(4):
        counter.update(80, visibility=0.1)
    # User comes back into frame and tries to finish the rep
    for angle in (100, 160):
        counter.update(angle, visibility=1.0)
    # Rep should not have counted because the in-flight rep was aborted
    assert counter.rep_count == 0


def test_exercise_phase_mappings_use_lift_semantic():
    """Squat/pushup are identity. Bicep curl swaps eccentric ↔ concentric."""
    assert SquatModule()._to_semantic("eccentric") == "eccentric"
    assert SquatModule()._to_semantic("concentric") == "concentric"
    assert PushupModule()._to_semantic("eccentric") == "eccentric"
    assert PushupModule()._to_semantic("concentric") == "concentric"
    # Curl: angle decreasing (mech eccentric) is the *concentric* (curl up).
    assert BicepCurlModule()._to_semantic("eccentric") == "concentric"
    assert BicepCurlModule()._to_semantic("concentric") == "eccentric"


def test_phase_display_text_is_per_exercise():
    assert SquatModule()._phase_display("eccentric") == "Lowering down"
    assert PushupModule()._phase_display("concentric") == "Pressing up"
    assert BicepCurlModule()._phase_display("concentric") == "Curling up"


def test_alternate_curl_total_reps_are_left_plus_right():
    module = AlternateBicepCurlModule()
    module._left_rep_counter.min_rep_duration = 0
    module._right_rep_counter.min_rep_duration = 0

    for angle in (170, 130, 50, 100, 165):
        _, done = module._left_rep_counter.update(angle, left_angle=angle, visibility=1.0)
        if done:
            module._left_rep_count += 1

    for angle in (170, 130, 50, 100, 165):
        _, done = module._right_rep_counter.update(angle, right_angle=angle, visibility=1.0)
        if done:
            module._right_rep_count += 1

    module.rep_count = module.left_reps + module.right_reps

    assert module.left_reps == 1
    assert module.right_reps == 1
    assert module.rep_count == 2
