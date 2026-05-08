"""
Hidden Markov Model for exercise state estimation.

Hidden states: IDLE, SQUAT, PUSHUP, CURL, ALT_CURL
Observations: discrete binary features derived from BodyFrame
Forward algorithm with log-space numerics to prevent underflow.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .features import BodyFrame, ViewEstimate


class ExState(IntEnum):
    IDLE = 0
    SQUAT = 1
    PUSHUP = 2
    CURL = 3
    ALT_CURL = 4


N_STATES = len(ExState)
STATE_NAMES = ["idle", "squat", "pushup", "bicep_curl", "alternate_bicep_curl"]


@dataclass
class HMMConfig:
    # Transition matrix prior: probability of staying in same state
    self_loop_prob: float = 0.97
    # Probability of transitioning to IDLE from any active state
    to_idle_prob: float = 0.01
    # Scale multiplier for emission Gaussian variance
    obs_variance_scale: float = 1.0


@dataclass
class HMMResult:
    """Posterior distribution over exercise states."""
    posterior: np.ndarray          # shape (5,) — P(state | obs_1..t)
    most_likely_state: ExState
    exercise_confidence: float     # posterior of dominant non-IDLE state
    state_name: str                # "squat", "pushup", etc. or "idle"


def _build_transition_matrix(cfg: HMMConfig) -> np.ndarray:
    """
    Build 5x5 transition matrix.
    High self-loop, small cross-exercise, IDLE↔exercise configurable.
    """
    A = np.full((N_STATES, N_STATES), 0.01)
    # Self-loops
    for i in range(N_STATES):
        A[i, i] = cfg.self_loop_prob
    # Transitions FROM active states TO idle
    for i in range(1, N_STATES):
        A[i, ExState.IDLE] = cfg.to_idle_prob
        A[ExState.IDLE, i] = (1.0 - cfg.self_loop_prob) / (N_STATES - 1)
    # Normalize rows
    row_sums = A.sum(axis=1, keepdims=True)
    A /= row_sums
    return A


N_FEATURES = 8


def _extract_obs_features(frame: BodyFrame, arm_asym_ema: float, elbow_flex_ema: float) -> np.ndarray:
    """
    Extract 8 observation features from BodyFrame.

    Features:
    0 - body_horizontal: body roughly horizontal (push-up)
    1 - knee_bent_no_curl: knees bent AND arms straight (pure squat signal)
    2 - elbow_flexion_active: min elbow angle <140 (arms bent)
    3 - hip_y_low: hips low in frame (squat regardless of knee angle)
    4 - arm_asymmetry_ema: EMA-smoothed |L_elbow - R_elbow| normalized.
        EMA survives the symmetric mid-rep frame in alt-curls where the
        instantaneous asymmetry collapses to zero.
    5 - vertical_stance: torso angle <30 from vertical
    6 - elbow_curl_no_knees: elbow flexion WITHOUT knee flexion (standing curl)
    7 - arm_phase_diff: signed product of L/R elbow velocities. +1 = both
        flexing/extending together (standard curl), -1 = opposite directions
        (alt curl), 0 = arms still (squat/pushup/idle). This signal does NOT
        vanish at mid-rep crossover — the velocities keep their opposite
        signs throughout.
    """
    angles = frame.angles
    left_knee = angles.get("left_knee", 180.0)
    right_knee = angles.get("right_knee", 180.0)
    left_elbow = angles.get("left_elbow", 180.0)
    right_elbow = angles.get("right_elbow", 180.0)
    torso = angles.get("torso_angle", 0.0)

    min_knee = min(left_knee, right_knee)
    min_elbow = min(left_elbow, right_elbow)

    knee_bent = min_knee < 120.0          # clearly bent (seated or deep squat)
    elbow_bent = min_elbow < 140.0        # arm curling

    # Squat: knees bent, arms straight
    knee_bent_no_curl = 1.0 if (knee_bent and not elbow_bent) else 0.0

    # Standing curl: elbow bent, knees straight
    elbow_curl_no_knees = 1.0 if (elbow_bent and not knee_bent) else 0.0

    # Hip low: hips in lower half of frame. Used to be ANDed with min_knee>120
    # which made it false during the actual squat motion (knees at 70-110)
    # — that bug penalized real squats. Drop the AND.
    hip_y_low = 1.0 if frame.hip_y > 0.55 else 0.0

    obs = np.array([
        1.0 if frame.is_horizontal else 0.0,
        knee_bent_no_curl,
        elbow_flex_ema,                          # smoothed; survives mid-rep extension
        hip_y_low,
        arm_asym_ema,
        1.0 if torso < 30.0 else 0.0,
        elbow_curl_no_knees,
        frame.arm_phase_diff,
    ], dtype=np.float64)
    return obs


# Emission probabilities: Gaussian means per state per feature
# Shape: (N_STATES, 8)
# Features: body_horiz | knee_bent_no_curl | elbow_flex | hip_low | arm_asym | vert_stance | elbow_no_knees | phase_diff
_EMISSION_MEANS = np.array([
    # horiz  kn_no_curl  elbow  hip_low  asym  vert  elbow_no_kn  phase
    [0.0,    0.0,        0.0,   0.0,     0.0,  0.5,  0.0,         0.0],   # IDLE
    [0.0,    0.95,       0.0,   0.85,    0.0,  0.6,  0.0,         0.0],   # SQUAT  (hip_low up to 0.85; vert relaxed: torso leans forward)
    [0.95,   0.0,        0.85,  0.4,     0.0,  0.1,  0.0,         0.0],   # PUSHUP
    [0.0,    0.0,        0.9,   0.0,     0.1,  0.9,  0.5,         0.5],   # CURL: arms in phase
    [0.0,    0.0,        0.85,  0.0,     0.75, 0.9,  0.5,        -0.7],   # ALT_CURL: arms anti-phase
], dtype=np.float64)

_EMISSION_VARS = np.full((N_STATES, N_FEATURES), 0.08, dtype=np.float64)
# Wider variance for features that are less discriminative
_EMISSION_VARS[:, 5] = 0.15   # vert_stance — can vary
_EMISSION_VARS[:, 4] = 0.12   # arm_asym baseline (slightly wider with EMA smoothing)
_EMISSION_VARS[ExState.SQUAT, 4] = 0.05    # squats: low asymmetry required
_EMISSION_VARS[ExState.PUSHUP, 4] = 0.05   # pushups: low asymmetry required
_EMISSION_VARS[ExState.ALT_CURL, 4] = 0.10 # alt curl: clear asymmetry required (wider — EMA dampens peaks)
# elbow_flexion_active for ALT_CURL: between alternating reps the curling
# arm is briefly nearly straight (both elbows >140°). Wider variance so
# those transition frames don't unseat the state.
_EMISSION_VARS[ExState.CURL, 2] = 0.15
_EMISSION_VARS[ExState.ALT_CURL, 2] = 0.20
# elbow_no_knees: wide variance for both CURL and ALT_CURL (both can be seated)
_EMISSION_VARS[ExState.CURL, 6] = 0.25
_EMISSION_VARS[ExState.ALT_CURL, 6] = 0.25
# knee_bent_no_curl: SQUAT must fire, others must not
_EMISSION_VARS[ExState.SQUAT, 1] = 0.04
_EMISSION_VARS[ExState.CURL, 1] = 0.04
_EMISSION_VARS[ExState.ALT_CURL, 1] = 0.04
# phase_diff: must discriminate CURL (+1) from ALT_CURL (-1). Wider so
# transient zero frames don't kill the state.
_EMISSION_VARS[ExState.CURL, 7] = 0.30
_EMISSION_VARS[ExState.ALT_CURL, 7] = 0.30
# Other states are agnostic to phase_diff (arms still anyway) — wide variance.
_EMISSION_VARS[ExState.IDLE, 7] = 0.5
_EMISSION_VARS[ExState.SQUAT, 7] = 0.5
_EMISSION_VARS[ExState.PUSHUP, 7] = 0.5


class ExerciseHMM:
    """
    Forward-algorithm HMM for exercise classification.
    Maintains running log-alpha (forward variable) across frames, plus
    an EMA of arm asymmetry (so the symmetric mid-rep frame in alt-curls
    doesn't collapse the ALT_CURL likelihood).
    """

    # EMA decay for arm-asymmetry feature. 0.8 = ~5-frame memory.
    _ASYM_EMA_DECAY = 0.8
    # EMA decay for elbow_flexion_active feature. Same span as asym.
    # Smooths over the brief moment between alternating curls when both
    # arms are nearly straight (elbow > 140°).
    _ELBOW_EMA_DECAY = 0.8
    # Threshold above which ALT_CURL gets a self-bias bonus on the next frame.
    _ALT_CURL_HOLD_THRESHOLD = 0.4
    # Log-emission bonus added to ALT_CURL when the slow-decay memory exceeds
    # the hold threshold. Equivalent to a stronger self-loop for ALT_CURL
    # only — narrow fix, doesn't make other states stickier.
    _ALT_CURL_HOLD_BONUS = 1.0
    # Slow-decay memory of recent ALT_CURL posterior. 0.97 ≈ 50-frame half-
    # life (~2.5s @ 20fps) so the hold bias survives the multi-second
    # one-arm-at-a-time transition where arm_phase_diff briefly collapses
    # to zero (only one arm moving). Without this, alt-curl detection
    # vanishes when the user lowers a weight on one arm.
    _ALT_CURL_MEM_DECAY = 0.97

    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()
        self._A = _build_transition_matrix(self.config)
        # log-alpha: log P(state_t, obs_1..t)
        self._log_alpha = np.log(np.full(N_STATES, 1.0 / N_STATES))
        self._emission_means = _EMISSION_MEANS.copy()
        self._emission_vars = _EMISSION_VARS * self.config.obs_variance_scale
        self._arm_asym_ema: float = 0.0
        self._elbow_flex_ema: float = 0.0
        self._prev_alt_curl_post: float = 0.0
        # Peak-tracking slow decay over recent ALT_CURL posterior.
        self._alt_curl_memory: float = 0.0

    def _log_emission(self, state: int, obs: np.ndarray) -> float:
        """Log probability of obs given state (Gaussian)."""
        mu = self._emission_means[state]
        var = self._emission_vars[state]
        log_p = -0.5 * np.sum((obs - mu) ** 2 / var + np.log(2 * np.pi * var))

        arm_asym = float(obs[4])    # feature 4 (EMA-smoothed)
        elbow_flex = float(obs[2])  # feature 2
        phase_diff = float(obs[7])  # feature 7

        if state == ExState.ALT_CURL:
            # Reward asymmetry AND anti-phase motion; either signal is enough
            # to keep ALT_CURL alive across crossover frames.
            log_p += 4.0 * arm_asym + 1.5 * elbow_flex * arm_asym
            log_p += 2.5 * max(0.0, -phase_diff)  # rewards phase_diff < 0
            # Self-bias hysteresis based on slow-decay memory of recent
            # ALT_CURL posterior. Survives multi-frame collapses (e.g. when
            # one arm holds while the other lowers, killing arm_phase_diff).
            if self._alt_curl_memory > self._ALT_CURL_HOLD_THRESHOLD:
                log_p += self._ALT_CURL_HOLD_BONUS * min(1.0, self._alt_curl_memory)
        elif state == ExState.CURL:
            # Both standing and seated curls: reward elbow flexion + in-phase
            log_p += 2.0 * elbow_flex
            log_p += 1.0 * max(0.0, phase_diff)   # rewards phase_diff > 0
        elif state == ExState.SQUAT:
            # Hard penalties: squats never have arm asymmetry or elbow flexion
            log_p -= 7.0 * arm_asym
            log_p -= 4.0 * elbow_flex
        elif state == ExState.PUSHUP:
            # Hard penalty: pushups never have arm asymmetry
            log_p -= 6.0 * arm_asym

        return float(log_p)

    def update(self, frame: BodyFrame) -> HMMResult:
        """
        Run one forward step.
        Returns posterior P(state | obs_1..t).
        """
        # Update arm-asymmetry EMA. Uses raw |L-R| degrees normalized by 40.
        left_elbow = frame.angles.get("left_elbow", 180.0)
        right_elbow = frame.angles.get("right_elbow", 180.0)
        instant_asym = min(1.0, abs(left_elbow - right_elbow) / 40.0)
        a = self._ASYM_EMA_DECAY
        self._arm_asym_ema = a * self._arm_asym_ema + (1.0 - a) * instant_asym

        # Update elbow-flexion EMA. Instant signal is "min elbow < 140".
        min_elbow = min(left_elbow, right_elbow)
        instant_elbow_flex = 1.0 if min_elbow < 140.0 else 0.0
        b = self._ELBOW_EMA_DECAY
        self._elbow_flex_ema = b * self._elbow_flex_ema + (1.0 - b) * instant_elbow_flex

        obs = _extract_obs_features(frame, self._arm_asym_ema, self._elbow_flex_ema)

        # Emission log-probabilities
        log_b = np.array([self._log_emission(s, obs) for s in range(N_STATES)])

        # Forward predict: log P(s_t | obs_1..t-1)
        log_A = np.log(self._A + 1e-12)
        # log-sum-exp over previous states
        log_predict = np.array([
            np.logaddexp.reduce(self._log_alpha + log_A[:, s])
            for s in range(N_STATES)
        ])

        # Update with observation
        log_alpha_new = log_predict + log_b

        # Normalize (subtract log-sum for numerical stability)
        log_z = np.logaddexp.reduce(log_alpha_new)
        self._log_alpha = log_alpha_new - log_z

        posterior = np.exp(self._log_alpha)
        posterior = np.clip(posterior, 0.0, 1.0)
        posterior /= posterior.sum()

        # Cache ALT_CURL posterior for next-frame self-bias and update the
        # slow-decay memory. Memory tracks the running maximum decayed at
        # _ALT_CURL_MEM_DECAY per frame, so a confident detection earlier
        # propagates forward even when the current posterior collapses.
        alt_post_now = float(posterior[ExState.ALT_CURL])
        self._prev_alt_curl_post = alt_post_now
        self._alt_curl_memory = max(
            alt_post_now,
            self._ALT_CURL_MEM_DECAY * self._alt_curl_memory,
        )

        # Most likely state
        ml_state = ExState(int(np.argmax(posterior)))

        # Exercise confidence: posterior of best non-IDLE state
        non_idle = posterior[1:]  # states 1..4
        if non_idle.max() > posterior[ExState.IDLE]:
            ex_conf = float(non_idle.max())
            best_ex_idx = int(np.argmax(non_idle)) + 1
            ml_state = ExState(best_ex_idx)
        else:
            ex_conf = 0.0
            ml_state = ExState.IDLE

        return HMMResult(
            posterior=posterior,
            most_likely_state=ml_state,
            exercise_confidence=ex_conf,
            state_name=STATE_NAMES[int(ml_state)],
        )

    def reset(self):
        self._log_alpha = np.log(np.full(N_STATES, 1.0 / N_STATES))
        self._arm_asym_ema = 0.0
        self._elbow_flex_ema = 0.0
        self._prev_alt_curl_post = 0.0
        self._alt_curl_memory = 0.0
