---
name: validate-websocket-contract
description: "Use when changing backend WebSocket payloads, response fields, exercise feedback schema, or state transitions consumed by the external Next.js PWA client."
---

# Validate WebSocket Contract

## Purpose
Provide a repeatable backend-first workflow to prevent breaking the WebSocket contract used by the external Next.js PWA frontend.

## Read First
- Protocol reference: [README.md](../../../README.md)
- WebSocket route and response creation: [backend/api/routes.py](../../../backend/api/routes.py)
- Session state machine outputs: [backend/state_machine/manager.py](../../../backend/state_machine/manager.py)
- Exercise result contract: [backend/exercises/base.py](../../../backend/exercises/base.py)
- Classifier/exercise switching behavior: [backend/exercises/classifier.py](../../../backend/exercises/classifier.py)

## When To Use
- Adding, renaming, or removing WebSocket response fields.
- Changing state names or exercise display behavior.
- Updating rep phase/count semantics.
- Modifying violations, corrections, confidence, or joint color outputs.

## Workflow
1. Identify the exact response fields and state outputs that changed.
2. Trace each field from producer to serializer in backend modules.
3. Verify required compatibility fields remain present and typed consistently.
4. Validate semantics, not only presence:
- rep_count progression is monotonic per session.
- rep_phase transitions are coherent for the active exercise.
- violations and corrections remain actionable strings.
- confidence remains in expected range.
5. If any intentional breaking change exists, update docs and call it out explicitly.

## Compatibility Guardrails
- Keep stable keys for client-facing payloads whenever possible.
- Prefer additive changes over destructive renames.
- If renaming is unavoidable, provide temporary alias fields for migration when feasible.
- Preserve response behavior expected by the external Next.js PWA client.

## Minimal Validation Commands
- Install backend deps:
  - pip install -r backend/requirements.txt
- Start backend:
  - cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
- Validate endpoints:
  - GET /
  - GET /api/health
- Validate one WebSocket roundtrip with realistic landmarks and confirm returned payload shape.

## Validation Checklist
- Response includes expected core fields for current state.
- No accidental field deletions or type changes.
- State transitions remain valid (idle/scanning/active behavior).
- Existing exercise modules still produce coherent feedback.
- README protocol section updated if response contract changed.

## Output Expectations For This Skill
When invoked, produce:
1. A short compatibility risk summary.
2. A list of changed fields with before/after semantics.
3. A validation report from endpoint and WebSocket checks.
4. Any required migration notes for the external Next.js PWA.
