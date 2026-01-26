"""Form Manager with state machine for exercise tracking."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Type
import time

from config.settings import settings
from exercises.base import BaseExercise, ExerciseResult, landmarks_to_dict, JointName
from exercises.classifier import ExerciseClassifier, ExerciseType, MotionAnalysis
from exercises.squat import SquatModule
from exercises.pushup import PushupModule
from exercises.bicep_curl import BicepCurlModule, AlternateBicepCurlModule


class SystemState(str, Enum):
    """System state machine states."""
    IDLE = "idle"  # No person detected
    SCANNING = "scanning"  # Person detected, waiting for movement
    ACTIVE = "active"  # Exercise identified, correction logic running


@dataclass
class FormManagerState:
    """Current state of the FormManager."""
    system_state: SystemState
    current_exercise: Optional[ExerciseType]
    exercise_result: Optional[ExerciseResult]
    motion_analysis: Optional[MotionAnalysis]
    time_in_state: float = 0.0
    frames_processed: int = 0


class FormManager:
    """
    Manages exercise detection, state transitions, and form correction.
    
    State Machine:
    - IDLE: No person detected in frame
    - SCANNING: Person detected, analyzing motion to identify exercise
    - ACTIVE: Exercise identified, running correction logic
    
    Features:
    - Automatic exercise recognition from motion patterns
    - Hot-swap of correction modules on exercise change
    - Confidence threshold for exercise lock-in
    """
    
    # Map exercise types to module classes
    EXERCISE_MODULES: dict[ExerciseType, Type[BaseExercise]] = {
        ExerciseType.SQUAT: SquatModule,
        ExerciseType.PUSHUP: PushupModule,
        ExerciseType.BICEP_CURL: BicepCurlModule,
        ExerciseType.ALTERNATE_BICEP_CURL: AlternateBicepCurlModule,
    }
    
    def __init__(self):
        self._state = SystemState.IDLE
        self._classifier = ExerciseClassifier()
        self._active_module: Optional[BaseExercise] = None
        self._current_exercise: Optional[ExerciseType] = None
        
        # Timing
        self._state_start_time = time.time()
        self._exercise_detection_start: Optional[float] = None
        self._pending_exercise: Optional[ExerciseType] = None
        
        # Stats
        self._frames_processed = 0
        self._last_result: Optional[ExerciseResult] = None
        self._last_analysis: Optional[MotionAnalysis] = None
    
    def process_frame(self, landmarks: list[dict]) -> FormManagerState:
        """
        Process a single frame of landmark data.
        
        Args:
            landmarks: Raw landmark list from MediaPipe (33 landmarks)
            
        Returns:
            FormManagerState with current system state and results
        """
        self._frames_processed += 1
        
        # Check for person presence
        if not self._has_visible_person(landmarks):
            self._transition_to(SystemState.IDLE)
            return self._create_state()
        
        # Add frame to classifier buffer
        self._classifier.add_frame(landmarks)
        
        # Handle state-specific logic
        if self._state == SystemState.IDLE:
            self._transition_to(SystemState.SCANNING)
        
        if self._state == SystemState.SCANNING:
            self._handle_scanning_state(landmarks)
        
        elif self._state == SystemState.ACTIVE:
            self._handle_active_state(landmarks)
        
        return self._create_state()
    
    def _has_visible_person(self, landmarks: list[dict]) -> bool:
        """Check if a person is visible in the frame."""
        if not landmarks or len(landmarks) < 33:
            return False
        
        # Check visibility of key landmarks
        key_indices = [11, 12, 23, 24]  # Shoulders and hips
        visible_count = sum(
            1 for i in key_indices 
            if i < len(landmarks) and landmarks[i].get("visibility", 0) > 0.5
        )
        
        return visible_count >= 3
    
    def _handle_scanning_state(self, landmarks: list[dict]) -> None:
        """Handle logic for SCANNING state."""
        # Analyze motion to identify exercise
        analysis = self._classifier.identify_exercise()
        self._last_analysis = analysis
        
        # Check if we can lock in an exercise
        if (analysis.exercise_type != ExerciseType.UNKNOWN and 
            analysis.confidence >= settings.CONFIDENCE_THRESHOLD and
            analysis.has_full_rep):
            
            # Lock in the exercise
            self._current_exercise = analysis.exercise_type
            self._classifier.lock_exercise(analysis.exercise_type)
            self._activate_module(analysis.exercise_type)
            self._transition_to(SystemState.ACTIVE)
    
    def _handle_active_state(self, landmarks: list[dict]) -> None:
        """Handle logic for ACTIVE state."""
        if not self._active_module:
            self._transition_to(SystemState.SCANNING)
            return
        
        # Process frame with active module
        self._last_result = self._active_module.process_frame(landmarks)
        
        # Check for exercise switch
        analysis = self._classifier.identify_exercise()
        self._last_analysis = analysis
        
        if analysis.exercise_type != self._current_exercise:
            # New exercise detected - track how long
            if self._pending_exercise != analysis.exercise_type:
                self._pending_exercise = analysis.exercise_type
                self._exercise_detection_start = time.time()
            else:
                detection_duration = time.time() - (self._exercise_detection_start or time.time())
                
                if self._classifier.should_switch_exercise(analysis.exercise_type, detection_duration):
                    self._hot_swap_module(analysis.exercise_type)
        else:
            # Same exercise, reset pending
            self._pending_exercise = None
            self._exercise_detection_start = None
    
    def _activate_module(self, exercise_type: ExerciseType) -> None:
        """Activate the correction module for an exercise type."""
        module_class = self.EXERCISE_MODULES.get(exercise_type)
        if module_class:
            self._active_module = module_class()
            self._current_exercise = exercise_type
    
    def _hot_swap_module(self, new_exercise: ExerciseType) -> None:
        """Hot-swap to a different exercise module without restarting."""
        # Store rep count from previous module if needed
        previous_reps = self._active_module.rep_count if self._active_module else 0
        
        # Activate new module
        self._activate_module(new_exercise)
        self._classifier.lock_exercise(new_exercise)
        
        # Reset pending
        self._pending_exercise = None
        self._exercise_detection_start = None
        
        # Clear result to show transition
        self._last_result = None
    
    def _transition_to(self, new_state: SystemState) -> None:
        """Transition to a new state."""
        if new_state != self._state:
            self._state = new_state
            self._state_start_time = time.time()
            
            # State-specific reset
            if new_state == SystemState.IDLE:
                self._classifier.reset()
                self._active_module = None
                self._current_exercise = None
                self._last_result = None
                self._pending_exercise = None
            
            elif new_state == SystemState.SCANNING:
                self._classifier.unlock_exercise()
    
    def _create_state(self) -> FormManagerState:
        """Create current state snapshot."""
        return FormManagerState(
            system_state=self._state,
            current_exercise=self._current_exercise,
            exercise_result=self._last_result,
            motion_analysis=self._last_analysis,
            time_in_state=time.time() - self._state_start_time,
            frames_processed=self._frames_processed
        )
    
    def get_exercise_name(self) -> str:
        """Get display name for current exercise."""
        if self._active_module:
            return self._active_module.name
        if self._current_exercise:
            return self._current_exercise.value.replace("_", " ").title()
        return "Scanning..."
    
    def get_state_display(self) -> str:
        """Get display text for current state."""
        if self._state == SystemState.IDLE:
            return "Waiting for person..."
        elif self._state == SystemState.SCANNING:
            return "Detecting exercise..."
        else:
            return f"Activity: {self.get_exercise_name()}"
    
    def reset(self) -> None:
        """Reset all state."""
        self._state = SystemState.IDLE
        self._classifier.reset()
        self._active_module = None
        self._current_exercise = None
        self._state_start_time = time.time()
        self._exercise_detection_start = None
        self._pending_exercise = None
        self._frames_processed = 0
        self._last_result = None
        self._last_analysis = None
    
    @property
    def state(self) -> SystemState:
        """Current system state."""
        return self._state
    
    @property
    def current_exercise(self) -> Optional[ExerciseType]:
        """Currently active exercise type."""
        return self._current_exercise
    
    @property
    def active_module(self) -> Optional[BaseExercise]:
        """Currently active exercise module."""
        return self._active_module
    
    @property
    def rep_count(self) -> int:
        """Current rep count from active module."""
        return self._active_module.rep_count if self._active_module else 0
