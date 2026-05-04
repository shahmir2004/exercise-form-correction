/**
 * Main application component.
 * Integrates video processing, pose detection, and form correction.
 */

import { useCallback, useState, useEffect, useRef } from 'react';
import { VideoPlayer } from './components/VideoPlayer';
import { ExerciseDisplay } from './components/ExerciseDisplay';
import { useVideoProcessor } from './hooks/useVideoProcessor';
import { usePoseStream, FormCorrectionResponse } from './hooks/usePoseStream';
import { PoseLandmark } from './pose';
import {
  STGCNClassifier,
  WINDOW as STGCN_WINDOW,
  N_JOINTS as STGCN_N_JOINTS,
  COORD_DIM as STGCN_COORD_DIM,
  KEY_JOINT_INDICES,
} from './pose/STGCNClassifier';
import { Activity, Wifi, WifiOff } from 'lucide-react';
import { API_BASE_URL, IS_DEVELOPMENT } from './config';

type PoseLibraryFrame = {
  landmarks: PoseLandmark[];
  timestamp: number;
};

function extractNormalisedFrame(
  landmarks: { x: number; y: number; z: number; visibility: number }[],
): Float32Array | null {
  if (landmarks.length < 33) return null;
  const lHip = landmarks[23];
  const rHip = landmarks[24];
  const lSho = landmarks[11];
  const rSho = landmarks[12];
  const hipX = (lHip.x + rHip.x) / 2;
  const hipY = (lHip.y + rHip.y) / 2;
  const hipZ = (lHip.z + rHip.z) / 2;
  const shoX = (lSho.x + rSho.x) / 2;
  const shoY = (lSho.y + rSho.y) / 2;
  const shoZ = (lSho.z + rSho.z) / 2;
  const torso = Math.sqrt(
    (shoX - hipX) ** 2 + (shoY - hipY) ** 2 + (shoZ - hipZ) ** 2,
  ) || 1e-6;

  const frame = new Float32Array(STGCN_N_JOINTS * STGCN_COORD_DIM);
  for (let i = 0; i < STGCN_N_JOINTS; i++) {
    const lm = landmarks[KEY_JOINT_INDICES[i]];
    frame[i * STGCN_COORD_DIM + 0] = (lm.x - hipX) / torso;
    frame[i * STGCN_COORD_DIM + 1] = (lm.y - hipY) / torso;
    frame[i * STGCN_COORD_DIM + 2] = (lm.z - hipZ) / torso;
  }
  return frame;
}

function App() {
  const [currentLandmarks, setCurrentLandmarks] = useState<PoseLandmark[] | null>(null);
  const [formResponse, setFormResponse] = useState<FormCorrectionResponse | null>(null);

  const [poseLibraryExercise, setPoseLibraryExercise] = useState('squat');
  const [poseLibraryMaxFrames, setPoseLibraryMaxFrames] = useState(180);
  const [poseLibraryMinVisibility, setPoseLibraryMinVisibility] = useState(0.3);
  const [poseLibraryAppend, setPoseLibraryAppend] = useState(false);
  const [poseLibraryCaptureFps, setPoseLibraryCaptureFps] = useState(15);
  const [poseLibraryStats, setPoseLibraryStats] = useState({ frames: 0, duration: 0 });
  const [poseLibraryStatus, setPoseLibraryStatus] = useState({
    state: 'idle',
    message: '',
    downloadUrl: '',
  });
  const [isPoseLibraryRecording, setIsPoseLibraryRecording] = useState(false);

  const poseLibraryFramesRef = useRef<PoseLibraryFrame[]>([]);
  const poseLibraryStartRef = useRef<number | null>(null);
  const poseLibraryLastCaptureRef = useRef<number>(0);

  const stgcnRef = useRef<STGCNClassifier | null>(null);
  const stgcnWindowRef = useRef<Float32Array[]>([]);
  const clientProbsRef = useRef<Record<string, number> | null>(null);

  // Video processor hook
  const {
    videoElement,
    isVideoReady,
    isPlaying,
    duration,
    currentTime,
    loadFile,
    startWebcam,
    stopSource,
    play,
    pause,
    seek,
    isPoseReady,
    isProcessing,
    isRealtime,
    isSeekable,
    error: videoError,
  } = useVideoProcessor({
    onPoseResult: (result) => {
      setCurrentLandmarks(result.landmarks);
    },
    onError: (error) => {
      console.error('Video processing error:', error);
    },
  });

  // WebSocket connection to backend - only connect when we start processing
  const {
    isConnected,
    isConnecting,
    sendLandmarks,
    connect,
    disconnect,
  } = usePoseStream({
    autoConnect: false,  // Don't auto-connect, we'll manage this ourselves
    onResponse: (response) => {
      setFormResponse(response);
    },
    onError: (error) => {
      console.error('WebSocket error:', error);
    },
  });
  
  // Connect/disconnect WebSocket based on processing state
  useEffect(() => {
    if (isProcessing && !isConnected && !isConnecting) {
      connect();
    } else if (!isProcessing && isConnected) {
      stgcnWindowRef.current = [];
      disconnect();
    }
  }, [isProcessing, isConnected, isConnecting, connect, disconnect]);

  useEffect(() => {
    const clf = new STGCNClassifier();
    stgcnRef.current = clf;
    clf.loadWeights('/stgcn_weights.json', '/stgcn_scaler.json').catch(err => {
      console.warn('ST-GCN weights failed to load:', err);
    });
  }, []);

  // Send landmarks to backend when they update — run ST-GCN first
  useEffect(() => {
    if (!currentLandmarks || !isConnected || !isProcessing) return;

    let probs: Record<string, number> | null = null;
    const clf = stgcnRef.current;
    if (clf?.isReady) {
      const frame = extractNormalisedFrame(currentLandmarks);
      if (frame) {
        stgcnWindowRef.current.push(frame);
        if (stgcnWindowRef.current.length > STGCN_WINDOW) {
          stgcnWindowRef.current.shift();
        }
        if (stgcnWindowRef.current.length === STGCN_WINDOW) {
          probs = clf.infer(stgcnWindowRef.current);
          if (probs) clientProbsRef.current = probs;
        }
      }
    }

    sendLandmarks(currentLandmarks, performance.now(), probs);
  }, [currentLandmarks, isConnected, isProcessing, sendLandmarks]);

  const updatePoseLibraryStats = useCallback(() => {
    const start = poseLibraryStartRef.current;
    const duration = start ? (performance.now() - start) / 1000 : 0;
    setPoseLibraryStats({
      frames: poseLibraryFramesRef.current.length,
      duration,
    });
  }, []);

  const startPoseLibraryRecording = useCallback(() => {
    poseLibraryFramesRef.current = [];
    poseLibraryStartRef.current = performance.now();
    poseLibraryLastCaptureRef.current = 0;
    setPoseLibraryStatus({ state: 'recording', message: 'Recording...', downloadUrl: '' });
    setPoseLibraryStats({ frames: 0, duration: 0 });
    setIsPoseLibraryRecording(true);
  }, []);

  const stopPoseLibraryRecording = useCallback(() => {
    setIsPoseLibraryRecording(false);
    updatePoseLibraryStats();
    setPoseLibraryStatus((prev) => ({
      ...prev,
      state: prev.state === 'recording' ? 'idle' : prev.state,
      message: prev.state === 'recording' ? 'Recording stopped.' : prev.message,
    }));
  }, [updatePoseLibraryStats]);

  const clearPoseLibraryRecording = useCallback(() => {
    poseLibraryFramesRef.current = [];
    poseLibraryStartRef.current = null;
    poseLibraryLastCaptureRef.current = 0;
    setPoseLibraryStats({ frames: 0, duration: 0 });
    setPoseLibraryStatus({ state: 'idle', message: '', downloadUrl: '' });
  }, []);

  const exportPoseLibrary = useCallback(async () => {
    const frames = poseLibraryFramesRef.current;
    if (frames.length === 0) {
      setPoseLibraryStatus({ state: 'error', message: 'No frames captured.', downloadUrl: '' });
      return;
    }

    setPoseLibraryStatus({ state: 'saving', message: 'Saving pose library...', downloadUrl: '' });
    try {
      const response = await fetch(`${API_BASE_URL}/api/pose-library/record`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exercise: poseLibraryExercise,
          frames,
          max_frames: poseLibraryMaxFrames,
          min_visibility: poseLibraryMinVisibility,
          append: poseLibraryAppend,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        const detail = payload?.detail || 'Failed to save pose library.';
        throw new Error(detail);
      }

      setPoseLibraryStatus({
        state: 'done',
        message: `Saved ${payload.embedding_count} embeddings for ${payload.exercise}.`,
        downloadUrl: payload.download_url || '',
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save pose library.';
      setPoseLibraryStatus({ state: 'error', message, downloadUrl: '' });
    }
  }, [
    poseLibraryAppend,
    poseLibraryExercise,
    poseLibraryMaxFrames,
    poseLibraryMinVisibility,
  ]);

  useEffect(() => {
    if (!isPoseLibraryRecording || !currentLandmarks) {
      return;
    }

    if (currentLandmarks.length !== 33) {
      return;
    }

    const captureIntervalMs = 1000 / Math.max(1, poseLibraryCaptureFps);
    const now = performance.now();

    if (now - poseLibraryLastCaptureRef.current < captureIntervalMs) {
      return;
    }

    poseLibraryLastCaptureRef.current = now;
    const landmarksCopy = currentLandmarks.map((lm) => ({
      x: lm.x,
      y: lm.y,
      z: lm.z,
      visibility: lm.visibility,
    }));

    poseLibraryFramesRef.current.push({
      landmarks: landmarksCopy,
      timestamp: now,
    });

    if (poseLibraryFramesRef.current.length % 10 === 0) {
      updatePoseLibraryStats();
    }
  }, [
    currentLandmarks,
    isPoseLibraryRecording,
    poseLibraryCaptureFps,
    updatePoseLibraryStats,
  ]);

  useEffect(() => {
    if (isPoseLibraryRecording && !isProcessing) {
      stopPoseLibraryRecording();
    }
  }, [isPoseLibraryRecording, isProcessing, stopPoseLibraryRecording]);

  const handleFileSelect = useCallback(async (file: File) => {
    await loadFile(file);
  }, [loadFile]);

  const handleStartCamera = useCallback(async () => {
    await startWebcam({ facingMode: 'user' });
    await play();
  }, [startWebcam, play]);

  const handleStopCamera = useCallback(() => {
    stopSource();
    setCurrentLandmarks(null);
    setFormResponse(null);
  }, [stopSource]);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-500" />
            <h1 className="text-xl font-bold">Exercise Form Correction</h1>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Connection status */}
            <div className="flex items-center gap-2 text-sm">
              {isConnected ? (
                <>
                  <Wifi className="w-4 h-4 text-green-500" />
                  <span className="text-green-400">Connected</span>
                </>
              ) : isConnecting ? (
                <>
                  <Wifi className="w-4 h-4 text-yellow-500 animate-pulse" />
                  <span className="text-yellow-400">Connecting...</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-4 h-4 text-red-500" />
                  <span className="text-red-400">Disconnected</span>
                </>
              )}
            </div>

            {/* Pose detector status */}
            <div className="text-sm text-gray-400">
              Pose: {isPoseReady ? (
                <span className="text-green-400">Ready</span>
              ) : (
                <span className="text-yellow-400">Loading...</span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Video player - takes 2 columns */}
          <div className="lg:col-span-2">
            <VideoPlayer
              videoElement={videoElement}
              isPlaying={isPlaying}
              currentTime={currentTime}
              duration={duration}
              isRealtime={isRealtime}
              isSeekable={isSeekable}
              landmarks={currentLandmarks}
              jointColors={formResponse?.joint_colors || {}}
              onPlay={play}
              onPause={pause}
              onSeek={seek}
              onFileSelect={handleFileSelect}
              onStartCamera={handleStartCamera}
              onStopCamera={handleStopCamera}
            />

            {/* Processing status */}
            {isVideoReady && (
              <div className="mt-4 flex items-center gap-4 text-sm text-gray-400">
                <span>
                  Processing: {isProcessing ? (
                    <span className="text-green-400">Active</span>
                  ) : (
                    <span className="text-gray-500">Paused</span>
                  )}
                </span>
                {videoError && (
                  <span className="text-red-400">Error: {videoError.message}</span>
                )}
              </div>
            )}
          </div>

          {/* Sidebar - Exercise info */}
          <div className="space-y-6">
            {/* Exercise display */}
            <ExerciseDisplay
              response={formResponse}
              isConnected={isConnected}
            />

            {/* Instructions */}
            <div className="bg-gray-800 rounded-xl p-4">
              <h3 className="text-lg font-semibold mb-3">How to Use</h3>
              <ol className="space-y-2 text-sm text-gray-400 list-decimal list-inside">
                <li>Click Start Camera to open your device camera</li>
                <li>Or upload a video and press play</li>
                <li>The system will detect your exercise automatically</li>
                <li>Watch for form corrections in real-time</li>
                <li>Green skeleton = good form</li>
                <li>Red skeleton = needs correction</li>
              </ol>
            </div>

            {/* Supported exercises */}
            <div className="bg-gray-800 rounded-xl p-4">
              <h3 className="text-lg font-semibold mb-3">Supported Exercises</h3>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500" />
                  <span className="text-gray-300">Squats</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500" />
                  <span className="text-gray-300">Push-ups</span>
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500" />
                  <span className="text-gray-300">Bicep Curls</span>
                </li>
              </ul>
            </div>

            {/* Pose library export (development only) */}
            {IS_DEVELOPMENT && (
              <div className="bg-gray-800 rounded-xl p-4">
                <h3 className="text-lg font-semibold mb-3">Pose Library Export</h3>
                <div className="space-y-3 text-sm">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Exercise</label>
                    <select
                      value={poseLibraryExercise}
                      onChange={(e) => setPoseLibraryExercise(e.target.value)}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                      disabled={isPoseLibraryRecording}
                    >
                      <option value="squat">squat</option>
                      <option value="pushup">pushup</option>
                      <option value="bicep_curl">bicep_curl</option>
                      <option value="alternate_bicep_curl">alternate_bicep_curl</option>
                      <option value="idle">idle</option>
                    </select>
                  </div>

                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Capture FPS</label>
                      <input
                        type="number"
                        min={5}
                        max={60}
                        value={poseLibraryCaptureFps}
                        onChange={(e) => setPoseLibraryCaptureFps(Number(e.target.value))}
                        className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-white"
                        disabled={isPoseLibraryRecording}
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Max Frames</label>
                      <input
                        type="number"
                        min={30}
                        max={2000}
                        value={poseLibraryMaxFrames}
                        onChange={(e) => setPoseLibraryMaxFrames(Number(e.target.value))}
                        className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-white"
                        disabled={isPoseLibraryRecording}
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Min Vis</label>
                      <input
                        type="number"
                        step={0.05}
                        min={0}
                        max={1}
                        value={poseLibraryMinVisibility}
                        onChange={(e) => setPoseLibraryMinVisibility(Number(e.target.value))}
                        className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-white"
                        disabled={isPoseLibraryRecording}
                      />
                    </div>
                  </div>

                  <label className="flex items-center gap-2 text-xs text-gray-400">
                    <input
                      type="checkbox"
                      checked={poseLibraryAppend}
                      onChange={(e) => setPoseLibraryAppend(e.target.checked)}
                      disabled={isPoseLibraryRecording}
                    />
                    Append to existing
                  </label>

                  <div className="text-xs text-gray-400">
                    Frames: {poseLibraryStats.frames} | Duration: {poseLibraryStats.duration.toFixed(1)}s
                  </div>

                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={startPoseLibraryRecording}
                      disabled={isPoseLibraryRecording || !isProcessing}
                      className="px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 text-white rounded-md"
                    >
                      Start
                    </button>
                    <button
                      onClick={stopPoseLibraryRecording}
                      disabled={!isPoseLibraryRecording}
                      className="px-3 py-1.5 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-700 text-white rounded-md"
                    >
                      Stop
                    </button>
                    <button
                      onClick={exportPoseLibrary}
                      disabled={isPoseLibraryRecording || poseLibraryStats.frames === 0}
                      className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white rounded-md"
                    >
                      Export
                    </button>
                    <button
                      onClick={clearPoseLibraryRecording}
                      disabled={isPoseLibraryRecording || poseLibraryStats.frames === 0}
                      className="px-3 py-1.5 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-700 text-white rounded-md"
                    >
                      Clear
                    </button>
                  </div>

                  {!isProcessing && (
                    <div className="text-xs text-gray-500">
                      Start camera or play a video to enable recording.
                    </div>
                  )}

                  {poseLibraryStatus.message && (
                    <div
                      className={`text-xs ${
                        poseLibraryStatus.state === 'error'
                          ? 'text-red-400'
                          : poseLibraryStatus.state === 'done'
                          ? 'text-green-400'
                          : 'text-gray-400'
                      }`}
                    >
                      {poseLibraryStatus.message}
                          {poseLibraryStatus.downloadUrl && (
                            <a
                              className="ml-2 text-blue-400 hover:text-blue-300 underline"
                              href={`${API_BASE_URL}${poseLibraryStatus.downloadUrl}`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              Download JSON
                            </a>
                          )}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Debug info (development only) */}
            {IS_DEVELOPMENT && formResponse && (
              <div className="bg-gray-800 rounded-xl p-4">
                <h3 className="text-lg font-semibold mb-3">Debug Info</h3>
                <pre className="text-xs text-gray-400 overflow-auto max-h-40">
                  {JSON.stringify(formResponse, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 mt-8">
        <div className="max-w-7xl mx-auto px-4 py-4 text-center text-sm text-gray-400">
          Exercise Form Correction System - Powered by MediaPipe
        </div>
      </footer>
    </div>
  );
}

export default App;
