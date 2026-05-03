/**
 * React hook for managing video source and pose detection pipeline.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { VideoSource } from '../video';
import { FileVideoSource } from '../video/FileVideoSource';
import { WebcamSource, WebcamConfig } from '../video/WebcamSource';
import { PoseDetector, PoseResult, MotionBuffer } from '../pose';

export interface UseVideoProcessorOptions {
  onPoseResult?: (result: PoseResult) => void;
  onError?: (error: Error) => void;
  targetFps?: number;
  bufferSize?: number;
}

export interface UseVideoProcessorReturn {
  // Video source
  videoElement: HTMLVideoElement | null;
  isVideoReady: boolean;
  isPlaying: boolean;
  duration: number;
  currentTime: number;
  isRealtime: boolean;
  isSeekable: boolean;
  
  // Controls
  loadFile: (file: File) => Promise<void>;
  loadUrl: (url: string) => Promise<void>;
  startWebcam: (config?: WebcamConfig) => Promise<void>;
  stopSource: () => void;
  play: () => Promise<void>;
  pause: () => void;
  seek: (time: number) => void;
  
  // Pose detection
  isPoseReady: boolean;
  isProcessing: boolean;
  startProcessing: () => void;
  stopProcessing: () => void;
  
  // Motion buffer
  motionBuffer: MotionBuffer;
  
  // Error
  error: Error | null;
}

export function useVideoProcessor(options: UseVideoProcessorOptions = {}): UseVideoProcessorReturn {
  const {
    onPoseResult,
    onError,
    targetFps = 30,
    bufferSize = 60,
  } = options;

  // Video state
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isRealtime, setIsRealtime] = useState(false);
  const [isSeekable, setIsSeekable] = useState(false);

  // Pose state
  const [isPoseReady, setIsPoseReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // Refs
  const videoSourceRef = useRef<VideoSource | null>(null);
  const poseDetectorRef = useRef<PoseDetector | null>(null);
  const motionBufferRef = useRef<MotionBuffer>(new MotionBuffer(bufferSize));
  const animationFrameRef = useRef<number | null>(null);
  const lastProcessTimeRef = useRef<number>(0);
  const isProcessingRef = useRef(false);
  
  // Store callbacks in refs to avoid recreating effects
  const callbacksRef = useRef({ onPoseResult, onError });
  useEffect(() => {
    callbacksRef.current = { onPoseResult, onError };
  }, [onPoseResult, onError]);

  // Initialize pose detector ONCE on mount
  useEffect(() => {
    let mounted = true;
    
    // Don't reinitialize if we already have a detector
    if (poseDetectorRef.current) {
      return;
    }
    
    const detector = new PoseDetector();
    poseDetectorRef.current = detector;

    detector.onResults((result) => {
      motionBufferRef.current.addFrame(result.landmarks, result.timestamp);
      callbacksRef.current.onPoseResult?.(result);
    });

    // Initialize the detector asynchronously
    detector.initialize()
      .then(() => {
        if (mounted) {
          console.log('Pose detector ready');
          setIsPoseReady(true);
        }
      })
      .catch((err) => {
        console.error('Failed to initialize pose detector:', err);
        if (mounted) {
          setError(err instanceof Error ? err : new Error('Failed to initialize pose detector'));
        }
      });

    return () => {
      mounted = false;
      if (poseDetectorRef.current) {
        poseDetectorRef.current.close();
        poseDetectorRef.current = null;
      }
    };
  }, []); // Empty deps - only run once on mount

  // Handle video time updates
  useEffect(() => {
    if (!videoElement) return;

    const handleTimeUpdate = () => {
      setCurrentTime(videoElement.currentTime);
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => {
      setIsPlaying(false);
      stopProcessingInternal();
    };

    videoElement.addEventListener('timeupdate', handleTimeUpdate);
    videoElement.addEventListener('play', handlePlay);
    videoElement.addEventListener('pause', handlePause);
    videoElement.addEventListener('ended', handleEnded);

    return () => {
      videoElement.removeEventListener('timeupdate', handleTimeUpdate);
      videoElement.removeEventListener('play', handlePlay);
      videoElement.removeEventListener('pause', handlePause);
      videoElement.removeEventListener('ended', handleEnded);
    };
  }, [videoElement]);

  const initializeSource = useCallback(async (source: VideoSource) => {
    setError(null);
    setIsVideoReady(false);
    stopProcessingInternal();

    // Clean up previous source
    if (videoSourceRef.current) {
      videoSourceRef.current.stop();
    }

    videoSourceRef.current = source;

    const element = await source.initialize();
    const meta = source.getMetadata();

    setVideoElement(element);
    setDuration(Number.isFinite(meta.duration) ? meta.duration : 0);
    setIsRealtime(meta.isRealtime);
    setIsSeekable(source.isSeekable);
    setIsVideoReady(true);
    setCurrentTime(0);
    motionBufferRef.current.clear();
  }, [stopProcessingInternal]);

  const loadFile = useCallback(async (file: File) => {
    try {
      await initializeSource(FileVideoSource.fromFile(file));
    } catch (e) {
      const err = e instanceof Error ? e : new Error('Failed to load video');
      setError(err);
      callbacksRef.current.onError?.(err);
    }
  }, [initializeSource]);

  const loadUrl = useCallback(async (url: string) => {
    try {
      await initializeSource(FileVideoSource.fromUrl(url));
    } catch (e) {
      const err = e instanceof Error ? e : new Error('Failed to load video');
      setError(err);
      callbacksRef.current.onError?.(err);
    }
  }, [initializeSource]);

  const startWebcam = useCallback(async (config?: WebcamConfig) => {
    try {
      await initializeSource(new WebcamSource(config));
    } catch (e) {
      const err = e instanceof Error ? e : new Error('Failed to start webcam');
      setError(err);
      callbacksRef.current.onError?.(err);
    }
  }, [initializeSource]);

  const stopSource = useCallback(() => {
    stopProcessingInternal();
    if (videoSourceRef.current) {
      videoSourceRef.current.stop();
      videoSourceRef.current = null;
    }
    setVideoElement(null);
    setIsVideoReady(false);
    setIsPlaying(false);
    setDuration(0);
    setCurrentTime(0);
    setIsRealtime(false);
    setIsSeekable(false);
  }, [stopProcessingInternal]);

  const play = useCallback(async () => {
    if (videoSourceRef.current) {
      await videoSourceRef.current.play();
    }
  }, []);

  const pause = useCallback(() => {
    if (videoSourceRef.current) {
      videoSourceRef.current.pause();
    }
  }, []);

  const seek = useCallback((time: number) => {
    if (videoSourceRef.current) {
      videoSourceRef.current.seek(time);
      setCurrentTime(time);
    }
  }, []);

  // Use a ref-based processing loop that doesn't depend on state
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const targetFpsRef = useRef(targetFps);
  
  // Keep refs in sync
  useEffect(() => {
    videoElementRef.current = videoElement;
  }, [videoElement]);
  
  useEffect(() => {
    targetFpsRef.current = targetFps;
  }, [targetFps]);

  const processFrame = useCallback(() => {
    // Use refs to avoid stale closures
    if (!videoElementRef.current || !poseDetectorRef.current?.ready || !isProcessingRef.current) {
      return;
    }

    const now = performance.now();
    const frameInterval = 1000 / targetFpsRef.current;

    if (now - lastProcessTimeRef.current >= frameInterval) {
      lastProcessTimeRef.current = now;
      poseDetectorRef.current.processFrame(videoElementRef.current);
    }

    animationFrameRef.current = requestAnimationFrame(processFrame);
  }, []); // No dependencies - uses refs

  const stopProcessingInternal = useCallback(() => {
    isProcessingRef.current = false;
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    setIsProcessing(false);
  }, []);

  const startProcessing = useCallback(() => {
    if (!isPoseReady || !videoElement) {
      console.warn('Cannot start processing: detector or video not ready');
      return;
    }

    if (isProcessingRef.current) {
      return; // Already processing
    }

    isProcessingRef.current = true;
    setIsProcessing(true);
    lastProcessTimeRef.current = performance.now();
    animationFrameRef.current = requestAnimationFrame(processFrame);
  }, [isPoseReady, videoElement, processFrame]);

  const stopProcessing = useCallback(() => {
    stopProcessingInternal();
  }, [stopProcessingInternal]);

  // Start processing when video plays
  useEffect(() => {
    if (isPlaying && isPoseReady && videoElement) {
      startProcessing();
    } else if (!isPlaying) {
      stopProcessingInternal();
    }
  }, [isPlaying, isPoseReady, videoElement, startProcessing, stopProcessingInternal]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopProcessingInternal();
      if (videoSourceRef.current) {
        videoSourceRef.current.stop();
      }
    };
  }, []);

  return {
    videoElement,
    isVideoReady,
    isPlaying,
    duration,
    currentTime,
    isRealtime,
    isSeekable,
    loadFile,
    loadUrl,
    startWebcam,
    stopSource,
    play,
    pause,
    seek,
    isPoseReady,
    isProcessing,
    startProcessing,
    stopProcessing,
    motionBuffer: motionBufferRef.current,
    error,
  };
}
