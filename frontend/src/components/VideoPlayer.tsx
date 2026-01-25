/**
 * Video player component with pose overlay and controls.
 */

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Play, Pause, RotateCcw, Upload } from 'lucide-react';
import { SkeletonOverlay } from './SkeletonOverlay';
import { PoseLandmark } from '../pose';

export interface VideoPlayerProps {
  videoElement: HTMLVideoElement | null;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  landmarks: PoseLandmark[] | null;
  jointColors?: Record<string, string>;
  onPlay: () => void;
  onPause: () => void;
  onSeek: (time: number) => void;
  onFileSelect: (file: File) => void;
  className?: string;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  videoElement,
  isPlaying,
  currentTime,
  duration,
  landmarks,
  jointColors = {},
  onPlay,
  onPause,
  onSeek,
  onFileSelect,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dimensions, setDimensions] = useState({ width: 1280, height: 720 });

  // Update dimensions when video loads
  useEffect(() => {
    if (videoElement) {
      const updateDimensions = () => {
        setDimensions({
          width: videoElement.videoWidth || 1280,
          height: videoElement.videoHeight || 720,
        });
      };

      videoElement.addEventListener('loadedmetadata', updateDimensions);
      updateDimensions();

      return () => {
        videoElement.removeEventListener('loadedmetadata', updateDimensions);
      };
    }
  }, [videoElement]);

  // Attach video element to container
  useEffect(() => {
    if (videoElement && videoContainerRef.current) {
      videoElement.style.width = '100%';
      videoElement.style.height = '100%';
      videoElement.style.objectFit = 'contain';
      
      // Clear and append
      videoContainerRef.current.innerHTML = '';
      videoContainerRef.current.appendChild(videoElement);
    }
  }, [videoElement]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const handleSeekChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    onSeek(time);
  }, [onSeek]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  return (
    <div className={`flex flex-col ${className}`}>
      {/* Video Container */}
      <div
        ref={containerRef}
        className="video-container bg-gray-900 rounded-xl overflow-hidden"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {!videoElement ? (
          // Upload placeholder
          <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400">
            <Upload className="w-16 h-16 mb-4" />
            <p className="text-lg font-medium">Drop a video file here</p>
            <p className="text-sm mt-1">or click to browse</p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="mt-4 px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              Select Video
            </button>
          </div>
        ) : (
          <>
            {/* Video element container */}
            <div ref={videoContainerRef} className="absolute inset-0" />
            
            {/* Skeleton overlay */}
            <SkeletonOverlay
              landmarks={landmarks}
              jointColors={jointColors}
              width={dimensions.width}
              height={dimensions.height}
            />
          </>
        )}
      </div>

      {/* Controls */}
      {videoElement && (
        <div className="mt-4 bg-gray-800 rounded-xl p-4">
          {/* Progress bar */}
          <div className="flex items-center gap-3 mb-3">
            <span className="text-sm text-gray-400 font-mono w-12">
              {formatTime(currentTime)}
            </span>
            <input
              type="range"
              min={0}
              max={duration || 100}
              value={currentTime}
              onChange={handleSeekChange}
              className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                         [&::-webkit-slider-thumb]:appearance-none
                         [&::-webkit-slider-thumb]:w-4
                         [&::-webkit-slider-thumb]:h-4
                         [&::-webkit-slider-thumb]:bg-blue-500
                         [&::-webkit-slider-thumb]:rounded-full
                         [&::-webkit-slider-thumb]:cursor-pointer"
            />
            <span className="text-sm text-gray-400 font-mono w-12">
              {formatTime(duration)}
            </span>
          </div>

          {/* Buttons */}
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={() => onSeek(0)}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title="Restart"
            >
              <RotateCcw className="w-5 h-5" />
            </button>

            <button
              onClick={isPlaying ? onPause : onPlay}
              className="p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-full transition-colors"
            >
              {isPlaying ? (
                <Pause className="w-6 h-6" />
              ) : (
                <Play className="w-6 h-6 ml-0.5" />
              )}
            </button>

            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title="Load new video"
            >
              <Upload className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
};
