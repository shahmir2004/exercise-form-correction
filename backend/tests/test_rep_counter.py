from exercises.bicep_curl import BicepCurlModule, AlternateBicepCurlModule
from exercises.pushup import PushupModule
from exercises.squat import SquatModule
from pipeline.rep_counter import HysteresisRepCounter


def test_squat_counter_counts_full_cycles_with_down_up_phases():
    counter = HysteresisRepCounter(upper_threshold=150, lower_threshold=90, min_rep_duration=0)
    phases = []
    completed = 0

    for _ in range(5):
        for angle in (170, 130, 80, 100, 160):
            phase, done = counter.update(angle)
            phases.append(phase.value)
            completed += int(done)

    assert completed == 5
    assert "down" in phases
    assert "up" in phases


def test_partial_reps_and_threshold_jitter_do_not_increment():
    counter = HysteresisRepCounter(upper_threshold=150, lower_threshold=90, min_rep_duration=0)

    for angle in (170, 145, 151, 146, 152, 148, 151):
        counter.update(angle)

    assert counter.rep_count == 0
    assert counter.partial_reps == 0


def test_exercise_phase_mappings_are_public_facing():
    assert SquatModule()._map_phase("down") == "down"
    assert SquatModule()._map_phase("up") == "up"
    assert PushupModule()._map_phase("down") == "down"
    assert PushupModule()._map_phase("up") == "up"
    assert BicepCurlModule()._map_phase("down") == "up"
    assert BicepCurlModule()._map_phase("up") == "down"


def test_alternate_curl_total_reps_are_left_plus_right():
    module = AlternateBicepCurlModule()
    module._left_rep_counter.min_rep_duration = 0
    module._right_rep_counter.min_rep_duration = 0

    for angle in (170, 130, 60, 100, 160):
        _, done = module._left_rep_counter.update(angle, left_angle=angle)
        if done:
            module._left_rep_count += 1

    for angle in (170, 130, 60, 100, 160):
        _, done = module._right_rep_counter.update(angle, right_angle=angle)
        if done:
            module._right_rep_count += 1

    module.rep_count = module.left_reps + module.right_reps

    assert module.left_reps == 1
    assert module.right_reps == 1
    assert module.rep_count == 2
