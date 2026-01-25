/**
 * Abstract video source interface for modular video input.
 * Supports both file uploads (MVP) and real-time webcam (production).
 */

export interface VideoSourceConfig {
  type: 'file' | 'webcam' | 'stream';
  source?: string | MediaStream;
  fps?: number;
  width?: number;
  height?: number;
}

export interface VideoMetadata {
  width: number;
  height: number;
  duration: number;
  fps: number;
  isRealtime: boolean;
}

export abstract class VideoSource {
  protected config: VideoSourceConfig;
  protected videoElement: HTMLVideoElement | null = null;
  protected isPlaying = false;

  constructor(config: VideoSourceConfig) {
    this.config = config;
  }

  /**
   * Initialize the video source.
   */
  abstract initialize(): Promise<HTMLVideoElement>;

  /**
   * Start playing/streaming.
   */
  abstract play(): Promise<void>;

  /**
   * Pause playback.
   */
  abstract pause(): void;

  /**
   * Stop and cleanup resources.
   */
  abstract stop(): void;

  /**
   * Seek to a specific time (for file sources).
   */
  abstract seek(time: number): void;

  /**
   * Get video metadata.
   */
  abstract getMetadata(): VideoMetadata;

  /**
   * Whether this is a real-time source.
   */
  abstract get isRealtime(): boolean;

  /**
   * Whether source supports seeking.
   */
  abstract get isSeekable(): boolean;

  /**
   * Get current playback time.
   */
  get currentTime(): number {
    return this.videoElement?.currentTime ?? 0;
  }

  /**
   * Get the video element.
   */
  get element(): HTMLVideoElement | null {
    return this.videoElement;
  }

  /**
   * Check if currently playing.
   */
  get playing(): boolean {
    return this.isPlaying;
  }
}
