---
name: add-exercise-module
description: "Use when adding a backend exercise module, wiring classification/state-machine integration, and validating rep counting plus WebSocket feedback output."
---

# Add Exercise Module

## Purpose
Add a new backend exercise detector that follows the existing plugin architecture and remains compatible with the external Next.js PWA client.

## Read First
- Architecture and protocol overview: [README.md](../../../README.md)
- Exercise contract: [backend/exercises/base.py](../../../backend/exercises/base.py)
- Reference implementation: [backend/exercises/squat.py](../../../backend/exercises/squat.py)
- Classifier integration: [backend/exercises/classifier.py](../../../backend/exercises/classifier.py)
- Runtime orchestration: [backend/state_machine/manager.py](../../../backend/state_machine/manager.py)

## Workflow
1. Create a new module in [backend/exercises/](../../../backend/exercises/) (for example, backend/exercises/lunge.py).
2. Implement a class extending BaseExercise.
3. Define name and required_joints properties.
4. Add angle helpers and phase detection tuned for that movement.
5. Implement check_form with:
- clear violations (what is wrong)
- actionable corrections (what to change)
- meaningful joint_colors and confidence
6. Configure or tune hysteresis rep counting via _create_rep_counter and update_rep_counter.
7. Register and wire detection in classifier/manager as needed.
8. Keep response field names stable for client compatibility.
9. Update [README.md](../../../README.md) if behavior or protocol details changed.

## Module Checklist
- Uses only required landmarks and handles low visibility safely.
- Keeps thresholds local to the module as constants.
- Avoids noisy phase flicker and false reps.
- Returns ExerciseResult with populated violations/corrections and rep metadata.
- Preserves compatibility with existing state and API contracts.

## Minimal Verification
- Start backend locally:
  - pip install -r backend/requirements.txt
  - cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
- Send sample landmark frames over WebSocket and verify:
  - exercise_display selection
  - rep_count and rep_phase transitions
  - corrections and joint_colors consistency
- Regression-check existing exercises still classify and count correctly.

## Common Pitfalls
- Breaking hysteresis thresholds and introducing rep jitter.
- Coupling module logic to one camera angle without visibility guards.
- Changing response keys without updating docs/consumers.
- Putting shared logic in one module that belongs in base utilities.

## Output Expectations For This Skill
When invoked, produce:
1. A concise plan of touched backend files.
2. The code changes for the new module and integration points.
3. A short validation report covering WebSocket behavior and rep counting stability.
