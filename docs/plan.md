## Plan: Backend Expansion to Production Grade

Deliver a backend-only roadmap that expands exercise coverage while hardening uptime, accuracy, and performance for the external Next.js PWA contract. Recommended approach is balanced execution: run reliability and accuracy tracks in parallel, with strict contract and regression gates before rollout.

**Steps**
1. Phase 1 - Reliability and Contract Baseline
1.1 Define measurable SLOs and acceptance thresholds before feature work: availability target, WebSocket error-rate budget, p95/p99 latency budget, and analysis-accuracy regression threshold.  
1.2 Add production observability primitives across backend runtime: structured logs, per-session correlation IDs, and service-level metrics for active WebSocket sessions, disconnect causes, frame processing latency, and upload failures. Depends on 1.1.  
1.3 Harden runtime safety controls for uptime: idle-session eviction, connection caps, heartbeat/last-activity tracking, graceful shutdown hooks, and safer exception handling in stream processing. Depends on 1.2.  
1.4 Lock backend contract compatibility policy for external Next.js PWA: additive field changes by default, explicit migration notes for intentional breaks, and contract snapshot tests as merge gates. Parallel with 1.3 after 1.2.

2. Phase 2 - Accuracy and Test Infrastructure
2.1 Establish backend test harness for unit, integration, and contract tests (including WebSocket roundtrip and upload lifecycle tests). Depends on 1.4.  
2.2 Create deterministic landmark-stream fixture sets for good-form and violation scenarios; wire fixture replay tests to classifier, state machine, and rep counter. Depends on 2.1.  
2.3 Add regression tests for rep semantics and state transitions: rep_count monotonicity, phase coherence, confidence bounds, and violation/correction quality checks. Depends on 2.2.  
2.4 Add upload integrity tests: init, chunk, status, complete, cancel, duplicate handling, interrupted recovery, and checksum validation. Parallel with 2.3 after 2.1.

3. Phase 3 - Exercise Expansion (First Wave)
3.1 Introduce a standardized exercise-module template and checklist based on BaseExercise contract (threshold locality, visibility guards, actionable corrections, rep-counter tuning). Depends on 2.2.  
3.2 Implement Lunge module and integrate classification/state-machine routing. Depends on 3.1.  
3.3 Implement Deadlift module and integrate classification/state-machine routing. Depends on 3.1. Parallel with 3.2.  
3.4 Implement Plank module and integrate classification/state-machine routing. Depends on 3.1. Parallel with 3.2.  
3.5 Implement Shoulder Press module and integrate classification/state-machine routing. Depends on 3.1. Parallel with 3.2.  
3.6 Tune classifier weights and transition logic for 8-exercise coexistence (existing + new) and verify no regression in squat/pushup/curl detection. Depends on 3.2-3.5.

4. Phase 4 - Analysis Quality Upgrades
4.1 Add cross-exercise ROM scoring and partial-rep detection normalization to reduce false positives and reward full movement quality. Depends on 3.6.  
4.2 Add asymmetry scoring and bilateral drift checks for unilateral/bilateral movements. Parallel with 4.1 after 3.6.  
4.3 Add tempo quality and consistency trend scoring per exercise set, while preserving existing response keys and client behavior expectations. Depends on 4.1.  
4.4 Calibrate confidence gating and smoothing parameters per exercise profile to improve accuracy under noisy/low-visibility frames. Depends on 4.2 and 4.3.

5. Phase 5 - Performance and Scalability
5.1 Optimize classifier hot path by reducing repeated full-buffer scans, introducing incremental displacement calculations, and frame-level compute throttling in SCANNING state. Depends on 2.2 and 3.6.  
5.2 Improve upload performance and safety with atomic assembly, background cleanup policies, and corruption detection hooks. Depends on 2.4.  
5.3 Introduce deployment scaling strategy: sticky-session baseline first, then optional external session state (Redis) for multi-instance failover if concurrency targets exceed single-instance limits. Depends on 5.1 and 5.2.

6. Phase 6 - Release Reliability and Uptime Operations
6.1 Add CI gates for contract tests, regression fixtures, and smoke tests before merge/deploy. Depends on 2.3 and 2.4.  
6.2 Run load and soak tests (WebSocket concurrency + upload concurrency), set alert thresholds, and enforce canary rollout with rollback criteria tied to SLO budgets. Depends on 5.3 and 6.1.  
6.3 Publish production runbook for incident response: degraded mode behavior, reconnect expectations, rollback sequence, and post-incident validation checklist. Depends on 6.2.

**Relevant files**
- [backend/main.py](backend/main.py) - app lifecycle, shutdown hooks, middleware-level reliability hardening.
- [backend/config/settings.py](backend/config/settings.py) - SLO-related config knobs, safety defaults, feature flags, and validation.
- [backend/api/routes.py](backend/api/routes.py) - WebSocket contract enforcement, session controls, response safety, health endpoints.
- [backend/api/upload.py](backend/api/upload.py) - upload integrity, atomic completion behavior, cleanup and performance safeguards.
- [backend/state_machine/manager.py](backend/state_machine/manager.py) - transition reliability, module orchestration, switch behavior hardening.
- [backend/exercises/base.py](backend/exercises/base.py) - module contract, shared quality fields, and consistent extension points.
- [backend/exercises/classifier.py](backend/exercises/classifier.py) - classification scaling, confidence logic, performance optimization.
- [backend/exercises/squat.py](backend/exercises/squat.py) - reference threshold and violation style for new exercise modules.
- [backend/exercises/pushup.py](backend/exercises/pushup.py) - reference upper-body phase and correction logic patterns.
- [backend/exercises/bicep_curl.py](backend/exercises/bicep_curl.py) - reference unilateral/bilateral checks and drift patterns.
- [backend/utils/rep_counter.py](backend/utils/rep_counter.py) - hysteresis semantics, rep-quality evolution, edge-case stability.
- [backend/utils/smoothing.py](backend/utils/smoothing.py) - noise-handling and low-visibility robustness tuning.
- [README.md](README.md) - authoritative API and WebSocket contract documentation updates.
- [BACKEND_STATUS.md](BACKEND_STATUS.md) - convert from implementation estimate to evidence-backed pass/fail status.

**Verification**
1. Baseline and post-change backend startup/health validation: install dependencies, run service locally, verify root and health endpoints.
2. WebSocket contract regression validation with realistic landmark streams: confirm response schema stability, state semantics, and monotonic rep behavior.
3. Exercise regression suite over fixture streams: existing exercises must remain stable while new exercises achieve target precision/recall thresholds.
4. Upload lifecycle integration validation: init, chunk, status, complete, cancel, interrupted-resume, duplicate-name handling, integrity failures.
5. Performance validation under load: concurrency, sustained throughput, frame processing latency, memory growth profile, and disconnect recovery behavior.
6. Canary release validation: alert health, error budget, and rollback trigger checks in staging then production ramp.

**Decisions**
- Priority model is Balanced: reliability/uptime and analysis accuracy move in parallel with gating tests.
- First exercise wave includes Lunge, Deadlift, Plank, and Shoulder Press.
- Scope is backend-first only; local frontend is treated as placeholder while preserving external Next.js PWA compatibility.
- 100% uptime intent is translated into strict SLO-driven operations and redundancy (practically near-100 with monitored failover and rapid rollback).

**Further Considerations**
1. Availability target recommendation: adopt 99.95% first, then raise to 99.99% after multi-instance and failover prove stable under soak tests.
2. Accuracy governance recommendation: define per-exercise minimum precision/recall and enforce as CI gating thresholds before enabling production flags.
3. Rollout recommendation: release new exercises behind feature flags and enable by cohort to control risk while collecting real-world telemetry.