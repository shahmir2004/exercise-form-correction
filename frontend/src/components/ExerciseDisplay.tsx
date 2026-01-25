/**
 * Exercise display component showing current activity and form feedback.
 */

import React from 'react';
import { FormCorrectionResponse } from '../hooks/usePoseStream';
import { 
  Activity, 
  CheckCircle, 
  AlertCircle, 
  AlertTriangle, 
  Repeat, 
  Loader,
  User
} from 'lucide-react';

export interface ExerciseDisplayProps {
  response: FormCorrectionResponse | null;
  isConnected: boolean;
  className?: string;
}

const StateIcon: React.FC<{ state: string }> = ({ state }) => {
  switch (state) {
    case 'idle':
      return <User className="w-5 h-5 text-gray-400" />;
    case 'scanning':
      return <Loader className="w-5 h-5 text-blue-500 animate-spin" />;
    case 'active':
      return <Activity className="w-5 h-5 text-green-500" />;
    default:
      return <Activity className="w-5 h-5 text-gray-400" />;
  }
};

const FormStatusIcon: React.FC<{ isValid: boolean; violations: string[] }> = ({ isValid, violations }) => {
  if (violations.length === 0) {
    return <CheckCircle className="w-5 h-5 text-green-500" />;
  }
  if (!isValid) {
    return <AlertCircle className="w-5 h-5 text-red-500" />;
  }
  return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
};

export const ExerciseDisplay: React.FC<ExerciseDisplayProps> = ({
  response,
  isConnected,
  className = '',
}) => {
  if (!isConnected) {
    return (
      <div className={`bg-gray-800 rounded-xl p-4 text-white ${className}`}>
        <div className="flex items-center gap-2 text-gray-400">
          <Loader className="w-5 h-5" />
          <span>Connecting to server...</span>
        </div>
      </div>
    );
  }

  if (!response) {
    return (
      <div className={`bg-gray-800 rounded-xl p-4 text-white ${className}`}>
        <div className="flex items-center gap-2 text-gray-400">
          <User className="w-5 h-5" />
          <span>Waiting for video...</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-xl p-4 text-white ${className}`}>
      {/* Activity Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <StateIcon state={response.state} />
          <span className="text-lg font-semibold">
            {response.exercise_display}
          </span>
        </div>
        
        {response.state === 'active' && (
          <div className="flex items-center gap-2 bg-gray-700 px-3 py-1 rounded-full">
            <Repeat className="w-4 h-4" />
            <span className="font-mono text-xl">{response.rep_count}</span>
            <span className="text-sm text-gray-400">reps</span>
          </div>
        )}
      </div>

      {/* Form Status */}
      {response.state === 'active' && (
        <div className="space-y-3">
          {/* Form Quality Indicator */}
          <div className="flex items-center gap-2">
            <FormStatusIcon isValid={response.is_rep_valid} violations={response.violations} />
            <span className={`font-medium ${
              response.is_rep_valid && response.violations.length === 0
                ? 'text-green-400'
                : response.is_rep_valid
                ? 'text-yellow-400'
                : 'text-red-400'
            }`}>
              {response.is_rep_valid && response.violations.length === 0
                ? 'Great form!'
                : response.is_rep_valid
                ? 'Minor adjustments needed'
                : 'Form needs correction'}
            </span>
          </div>

          {/* Correction Message */}
          {response.correction_message && (
            <div className="bg-gray-700/50 rounded-lg p-3">
              <p className="text-sm text-gray-300">
                💡 {response.correction_message}
              </p>
            </div>
          )}

          {/* Violations List */}
          {response.violations.length > 0 && (
            <div className="space-y-1">
              {response.violations.map((violation, index) => (
                <div
                  key={index}
                  className="flex items-center gap-2 text-sm text-red-400"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
                  {violation}
                </div>
              ))}
            </div>
          )}

          {/* Rep Phase Indicator */}
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span>Phase:</span>
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
              response.rep_phase === 'up' ? 'bg-green-500/20 text-green-400' :
              response.rep_phase === 'down' ? 'bg-blue-500/20 text-blue-400' :
              response.rep_phase === 'hold' ? 'bg-yellow-500/20 text-yellow-400' :
              'bg-gray-500/20 text-gray-400'
            }`}>
              {response.rep_phase.toUpperCase()}
            </span>
            
            <span className="ml-2">Confidence:</span>
            <span className="font-mono">
              {(response.confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      )}

      {/* Scanning State */}
      {response.state === 'scanning' && (
        <div className="text-sm text-gray-400">
          <p className="status-scanning">
            Analyzing your movement pattern...
          </p>
          <p className="mt-1">
            Perform one complete rep to begin tracking
          </p>
        </div>
      )}

      {/* Idle State */}
      {response.state === 'idle' && (
        <div className="text-sm text-gray-400">
          <p>Position yourself in front of the camera</p>
          <p className="mt-1">Make sure your full body is visible</p>
        </div>
      )}
    </div>
  );
};
