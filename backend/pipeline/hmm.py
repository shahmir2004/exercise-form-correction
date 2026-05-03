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
    self_loop_prob: float = 0.95
    # Probability of transitioning to IDLE from any active state
    to_idle_prob: float = 0.02
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
    Extract 6 binary/continuous observation features from BodyFrame.

    Features:
    0 - body_horizontal: is body roughly horizontal?
    1 - knee_flexion_active: knee angle <140 (legs bent)
    2 - elbow_flexion_active: min elbow angle <140 (arms bent)
    3 - hip_y_low: hips in lower half of frame (squatting)
    4 - arm_asymmetry: abs(left_elbow - right_elbow) > 20 (alt curl)
    5 - vertical_stance: torso angle <30 from vertical
    """
    angles = frame.angles
    left_knee = angles.get("left_knee", 180.0)
    right_knee = angles.get("right_knee", 180.0)
    left_elbow = angles.get("left_elbow", 180.0)
    right_elbow = angles.get("right_elbow", 180.0)
    torso = angles.get("torso_angle", 0.0)

    obs = np.array([
        1.0 if frame.is_horizontal else 0.0,
        1.0 if min(left_knee, right_knee) < 140.0 else 0.0,
        1.0 if min(left_elbow, right_elbow) < 140.0 else 0.0,
        1.0 if frame.hip_y > 0.55 else 0.0,   # hips lower in frame = higher y value
        min(1.0, abs(left_elbow - right_elbow) / 40.0),
        1.0 if torso < 30.0 else 0.0,
    ], dtype=np.float64)
    return obs


# Emission probabilities: Gaussian means per state per feature
# Shape: (N_STATES, 6)
_EMISSION_MEANS = np.array([
    # body_horiz  knee_flex  elbow_flex  hip_low  arm_asym  vert_stance
    [0.0,         0.0,       0.0,        0.0,     0.0,      0.5],   # IDLE
    [0.0,         0.9,       0.0,        0.7,     0.1,      0.8],   # SQUAT
    [0.95,        0.1,       0.8,        0.5,     0.1,      0.1],   # PUSHUP
    [0.0,         0.0,       0.9,        0.0,     0.1,      0.9],   # CURL
    # Alternate curls are identified primarily by arm asymmetry; a strong
    # asymmetry boost is applied in _log_emission so seated curls do not fall
    # back to the squat state.
    [0.0,         0.0,       0.85,       0.0,     0.7,      0.9],   # ALT_CURL
], dtype=np.float64)

_EMISSION_VARS = np.full((N_STATES, 6), 0.12, dtype=np.float64)
_EMISSION_VARS[:, 4] = 0.20  # arm asymmetry has higher variance
_EMISSION_VARS[ExState.ALT_CURL, 4] = 0.08  # alternate curls should have clear asymmetry


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

        # Arm asymmetry is the key cue for alternate curls and a strong
        # negative cue for squats. This keeps seated alternating curls from
        # collapsing into the squat state while preserving regular curl scores.
        arm_asym = float(obs[4])
        if state == ExState.ALT_CURL:
            log_p += 3.5 * arm_asym
        elif state == ExState.SQUAT:
            log_p -= 4.5 * arm_asym

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
