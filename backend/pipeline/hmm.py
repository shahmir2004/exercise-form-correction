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


def _extract_obs_features(frame: BodyFrame) -> np.ndarray:
    """
    Extract 7 observation features from BodyFrame.

    Features:
    0 - body_horizontal: body roughly horizontal (push-up)
    1 - knee_bent_no_curl: knees bent AND arms straight (pure squat signal)
    2 - elbow_flexion_active: min elbow angle <140 (arms bent)
    3 - hip_y_low_standing: hips low in frame AND knees relatively straight
    4 - arm_asymmetry: abs(left_elbow - right_elbow) normalized (alt curl)
    5 - vertical_stance: torso angle <30 from vertical
    6 - elbow_curl_no_knees: elbow flexion WITHOUT knee flexion (standing curl)
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

    # Hip low only meaningful for squats (standing, knees somewhat straight)
    hip_y_low_standing = 1.0 if (frame.hip_y > 0.55 and min_knee > 120.0) else 0.0

    obs = np.array([
        1.0 if frame.is_horizontal else 0.0,
        knee_bent_no_curl,
        1.0 if elbow_bent else 0.0,
        hip_y_low_standing,
        min(1.0, abs(left_elbow - right_elbow) / 40.0),
        1.0 if torso < 30.0 else 0.0,
        elbow_curl_no_knees,
    ], dtype=np.float64)
    return obs


# Emission probabilities: Gaussian means per state per feature
# Shape: (N_STATES, 7)
# Features: body_horiz | knee_bent_no_curl | elbow_flex | hip_low_standing | arm_asym | vert_stance | elbow_no_knees
_EMISSION_MEANS = np.array([
    # horiz  kn_no_curl  elbow  hip_low  asym  vert  elbow_no_kn
    [0.0,    0.0,        0.0,   0.0,     0.0,  0.5,  0.0],   # IDLE
    [0.0,    0.95,       0.0,   0.7,     0.0,  0.8,  0.0],   # SQUAT
    [0.95,   0.0,        0.85,  0.4,     0.0,  0.1,  0.0],   # PUSHUP
    [0.0,    0.0,        0.9,   0.0,     0.1,  0.9,  0.5],   # CURL (standing or seated — elbow_no_knees is don't-care)
    [0.0,    0.0,        0.85,  0.0,     0.75, 0.9,  0.5],   # ALT_CURL (may be seated so elbow_no_knees can be 0)
], dtype=np.float64)

_EMISSION_VARS = np.full((N_STATES, 7), 0.08, dtype=np.float64)
# Wider variance for features that are less discriminative
_EMISSION_VARS[:, 5] = 0.15   # vert_stance — can vary
_EMISSION_VARS[:, 4] = 0.10   # arm_asym baseline
_EMISSION_VARS[ExState.SQUAT, 4] = 0.03    # squats: near-zero asymmetry required
_EMISSION_VARS[ExState.PUSHUP, 4] = 0.03   # pushups: near-zero asymmetry required
_EMISSION_VARS[ExState.ALT_CURL, 4] = 0.05 # alt curl: clear asymmetry required
# elbow_no_knees: wide variance for both CURL and ALT_CURL (both can be seated)
_EMISSION_VARS[ExState.CURL, 6] = 0.25
_EMISSION_VARS[ExState.ALT_CURL, 6] = 0.25
# knee_bent_no_curl: SQUAT must fire, others must not
_EMISSION_VARS[ExState.SQUAT, 1] = 0.04
_EMISSION_VARS[ExState.CURL, 1] = 0.04
_EMISSION_VARS[ExState.ALT_CURL, 1] = 0.04


class ExerciseHMM:
    """
    Forward-algorithm HMM for exercise classification.
    Maintains running log-alpha (forward variable) across frames.
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()
        self._A = _build_transition_matrix(self.config)
        # log-alpha: log P(state_t, obs_1..t)
        self._log_alpha = np.log(np.full(N_STATES, 1.0 / N_STATES))
        self._emission_means = _EMISSION_MEANS.copy()
        self._emission_vars = _EMISSION_VARS * self.config.obs_variance_scale

    def _log_emission(self, state: int, obs: np.ndarray) -> float:
        """Log probability of obs given state (Gaussian)."""
        mu = self._emission_means[state]
        var = self._emission_vars[state]
        log_p = -0.5 * np.sum((obs - mu) ** 2 / var + np.log(2 * np.pi * var))

        arm_asym = float(obs[4])   # feature 4
        elbow_flex = float(obs[2]) # feature 2

        if state == ExState.ALT_CURL:
            # Reward asymmetry; seated alt-curls may have low asymmetry mid-rep
            log_p += 4.0 * arm_asym + 1.5 * elbow_flex * arm_asym
        elif state == ExState.CURL:
            # Both standing and seated curls: reward elbow flexion
            log_p += 2.0 * elbow_flex
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
        obs = _extract_obs_features(frame)

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
