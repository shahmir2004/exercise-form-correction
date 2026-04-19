"""
Advanced rep counting with hysteresis and state machine.
Based on Stack Overflow best practices for exercise rep counting.

Implements:
- Hysteresis to prevent phase flickering
- State machine for reliable phase transitions
- Partial rep detection
- Rep quality scoring
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import time


class RepPhase(str, Enum):
    """Phases of a repetition."""
    IDLE = "idle"           # Not in exercise position
    READY = "ready"         # In starting position
    ECCENTRIC = "down"      # Lowering/extending phase
    CONCENTRIC = "up"       # Lifting/contracting phase
    HOLD = "hold"           # Paused at top/bottom
    TRANSITION = "transition"  # Between phases


@dataclass
class RepQuality:
    """Quality metrics for a single rep."""
    form_score: float = 1.0        # 0-1, based on form violations
    depth_score: float = 1.0       # 0-1, how deep the rep went
    tempo_score: float = 1.0       # 0-1, was tempo controlled
    symmetry_score: float = 1.0    # 0-1, left/right balance
    
    @property
    def overall_score(self) -> float:
        """Calculate overall rep quality score."""
        return (
            self.form_score * 0.4 + 
            self.depth_score * 0.3 + 
            self.tempo_score * 0.15 + 
            self.symmetry_score * 0.15
        )


@dataclass
class RepData:
    """Data for a completed rep."""
    rep_number: int
    quality: RepQuality
    duration_seconds: float
    min_angle: float
    max_angle: float
    timestamp: float = field(default_factory=time.time)


class HysteresisRepCounter:
    """
    Rep counter with hysteresis to prevent false triggers.
    
    Uses two thresholds instead of one:
    - upper_threshold: Angle must exceed this to register "up" phase
    - lower_threshold: Angle must go below this to register "down" phase
    
    This creates a "dead zone" that prevents oscillation from noise.
    """
    
    def __init__(
        self,
        upper_threshold: float,
        lower_threshold: float,
        min_rep_duration: float = 0.5,  # Minimum seconds for valid rep
        max_rep_duration: float = 10.0,  # Maximum seconds for valid rep
        require_full_extension: bool = True
    ):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.min_rep_duration = min_rep_duration
        self.max_rep_duration = max_rep_duration
        self.require_full_extension = require_full_extension
        
        # State
        self._phase = RepPhase.IDLE
        self._rep_count = 0
        self._partial_rep_count = 0
        self._rep_start_time: Optional[float] = None
        self._min_angle_in_rep: float = 180.0
        self._max_angle_in_rep: float = 0.0
        self._angle_history: deque[float] = deque(maxlen=30)
        self._reps: list[RepData] = []
        self._form_violations: list[str] = []
        
        # For symmetry tracking
        self._left_angle: float = 0.0
        self._right_angle: float = 0.0
    
    def update(
        self, 
        angle: float, 
        left_angle: Optional[float] = None,
        right_angle: Optional[float] = None,
        form_violations: Optional[list[str]] = None
    ) -> tuple[RepPhase, bool]:
        """
        Update with new angle reading.
        
        Args:
            angle: Primary angle to track (avg of left/right)
            left_angle: Left side angle (for symmetry)
            right_angle: Right side angle (for symmetry)
            form_violations: List of form issues detected this frame
            
        Returns:
            Tuple of (current_phase, rep_just_completed)
        """
        self._angle_history.append(angle)
        self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
        self._max_angle_in_rep = max(self._max_angle_in_rep, angle)
        
        if left_angle is not None:
            self._left_angle = left_angle
        if right_angle is not None:
            self._right_angle = right_angle
        if form_violations:
            self._form_violations.extend(form_violations)
        
        current_time = time.time()
        rep_completed = False
        previous_phase = self._phase
        
        # State machine with hysteresis
        if self._phase == RepPhase.IDLE:
            # Transition to READY when angle is near starting position
            if angle > self.upper_threshold - 10:
                self._phase = RepPhase.READY
                self._rep_start_time = current_time
        
        elif self._phase == RepPhase.READY:
            # Transition to DOWN when angle drops below upper threshold
            if angle < self.upper_threshold:
                self._phase = RepPhase.ECCENTRIC
                self._rep_start_time = current_time
                self._min_angle_in_rep = angle
                self._max_angle_in_rep = angle
                self._form_violations = []
        
        elif self._phase == RepPhase.ECCENTRIC:
            # Track minimum angle reached
            # Transition to UP when angle goes below lower threshold
            if angle < self.lower_threshold:
                self._phase = RepPhase.CONCENTRIC
        
        elif self._phase == RepPhase.CONCENTRIC:
            # Transition back to READY/complete rep when exceeding upper threshold
            if angle > self.upper_threshold:
                rep_duration = current_time - (self._rep_start_time or current_time)
                
                # Validate rep
                if self._is_valid_rep(rep_duration):
                    self._rep_count += 1
                    rep_completed = True
                    
                    # Calculate quality
                    quality = self._calculate_rep_quality(rep_duration)
                    
                    self._reps.append(RepData(
                        rep_number=self._rep_count,
                        quality=quality,
                        duration_seconds=rep_duration,
                        min_angle=self._min_angle_in_rep,
                        max_angle=self._max_angle_in_rep
                    ))
                else:
                    # Partial rep
                    self._partial_rep_count += 1
                
                # Reset for next rep
                self._phase = RepPhase.READY
                self._min_angle_in_rep = 180.0
                self._max_angle_in_rep = 0.0
                self._rep_start_time = current_time
                self._form_violations = []
        
        # Timeout: if stuck in a phase too long, reset
        if self._rep_start_time and (current_time - self._rep_start_time) > self.max_rep_duration:
            if self._phase in [RepPhase.ECCENTRIC, RepPhase.CONCENTRIC]:
                self._phase = RepPhase.READY
                self._rep_start_time = current_time
        
        return self._phase, rep_completed
    
    def _is_valid_rep(self, duration: float) -> bool:
        """Check if rep meets validity criteria."""
        # Duration check
        if duration < self.min_rep_duration:
            return False
        if duration > self.max_rep_duration:
            return False
        
        # Depth check (must have gone low enough)
        if self.require_full_extension:
            rep_range = self._max_angle_in_rep - self._min_angle_in_rep
            if rep_range < 30:  # Minimum 30 degrees of movement
                return False
        
        return True
    
    def _calculate_rep_quality(self, duration: float) -> RepQuality:
        """Calculate quality metrics for completed rep."""
        # Form score: based on violations
        form_score = max(0.0, 1.0 - len(self._form_violations) * 0.15)
        
        # Depth score: how much range of motion
        expected_range = self.upper_threshold - self.lower_threshold
        actual_range = self._max_angle_in_rep - self._min_angle_in_rep
        depth_score = min(1.0, actual_range / max(expected_range, 1))
        
        # Tempo score: ideal rep is 2-4 seconds
        if 2.0 <= duration <= 4.0:
            tempo_score = 1.0
        elif duration < 1.0:
            tempo_score = 0.5
        elif duration > 6.0:
            tempo_score = 0.7
        else:
            tempo_score = 0.85
        
        # Symmetry score: left/right balance
        if self._left_angle > 0 and self._right_angle > 0:
            angle_diff = abs(self._left_angle - self._right_angle)
            symmetry_score = max(0.0, 1.0 - angle_diff / 30)
        else:
            symmetry_score = 1.0
        
        return RepQuality(
            form_score=form_score,
            depth_score=depth_score,
            tempo_score=tempo_score,
            symmetry_score=symmetry_score
        )
    
    @property
    def rep_count(self) -> int:
        """Get current rep count."""
        return self._rep_count
    
    @property
    def phase(self) -> RepPhase:
        """Get current phase."""
        return self._phase
    
    @property
    def phase_str(self) -> str:
        """Get phase as string."""
        return self._phase.value
    
    @property
    def partial_reps(self) -> int:
        """Get count of partial/invalid reps."""
        return self._partial_rep_count
    
    @property
    def average_quality(self) -> float:
        """Get average quality score across all reps."""
        if not self._reps:
            return 0.0
        return sum(r.quality.overall_score for r in self._reps) / len(self._reps)
    
    @property
    def last_rep(self) -> Optional[RepData]:
        """Get data for last completed rep."""
        return self._reps[-1] if self._reps else None
    
    def reset(self):
        """Reset counter state."""
        self._phase = RepPhase.IDLE
        self._rep_count = 0
        self._partial_rep_count = 0
        self._rep_start_time = None
        self._min_angle_in_rep = 180.0
        self._max_angle_in_rep = 0.0
        self._angle_history.clear()
        self._reps.clear()
        self._form_violations.clear()


class ExerciseRepCounter:
    """
    Factory for creating exercise-specific rep counters.
    """
    
    @staticmethod
    def create_bicep_curl_counter() -> HysteresisRepCounter:
        """Create counter for bicep curls."""
        return HysteresisRepCounter(
            upper_threshold=150,   # Arm extended
            lower_threshold=50,    # Arm fully bent
            min_rep_duration=0.8,
            max_rep_duration=8.0,
            require_full_extension=True
        )
    
    @staticmethod
    def create_squat_counter() -> HysteresisRepCounter:
        """Create counter for squats."""
        return HysteresisRepCounter(
            upper_threshold=160,   # Standing
            lower_threshold=90,    # Parallel
            min_rep_duration=1.0,
            max_rep_duration=10.0,
            require_full_extension=True
        )
    
    @staticmethod
    def create_pushup_counter() -> HysteresisRepCounter:
        """Create counter for push-ups."""
        return HysteresisRepCounter(
            upper_threshold=155,   # Arms extended
            lower_threshold=90,    # Arms bent
            min_rep_duration=0.8,
            max_rep_duration=8.0,
            require_full_extension=True
        )
