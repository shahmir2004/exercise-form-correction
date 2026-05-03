/**
 * Main application component.
 * Integrates video processing, pose detection, and form correction.
 */

import { useCallback, useState, useEffect } from 'react';
import { VideoPlayer } from './components/VideoPlayer';
import { ExerciseDisplay } from './components/ExerciseDisplay';
import { useVideoProcessor } from './hooks/useVideoProcessor';
import { usePoseStream, FormCorrectionResponse } from './hooks/usePoseStream';
import { PoseLandmark } from './pose';
import { Activity, Wifi, WifiOff } from 'lucide-react';

function App() {
  const [currentLandmarks, setCurrentLandmarks] = useState<PoseLandmark[] | null>(null);
  const [formResponse, setFormResponse] = useState<FormCorrectionResponse | null>(null);

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
      disconnect();
    }
  }, [isProcessing, isConnected, isConnecting, connect, disconnect]);

  // Send landmarks to backend when they update
  useEffect(() => {
    if (currentLandmarks && isConnected && isProcessing) {
      sendLandmarks(currentLandmarks, performance.now());
    }
  }, [currentLandmarks, isConnected, isProcessing, sendLandmarks]);

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

            {/* Debug info (development only) */}
            {import.meta.env.DEV && formResponse && (
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
