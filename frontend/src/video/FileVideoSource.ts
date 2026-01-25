/**
 * File-based video source for uploaded videos.
 * Used for MVP development and video analysis.
 */

import { VideoSource, VideoMetadata } from './VideoSource';

export class FileVideoSource extends VideoSource {
  private objectUrl: string | null = null;

  constructor(file: File | string) {
    super({
      type: 'file',
      source: typeof file === 'string' ? file : undefined,
    });

    if (file instanceof File) {
      this.objectUrl = URL.createObjectURL(file);
    } else {
      this.objectUrl = file;
    }
  }

  async initialize(): Promise<HTMLVideoElement> {
    return new Promise((resolve, reject) => {
      this.videoElement = document.createElement('video');
      this.videoElement.crossOrigin = 'anonymous';
      this.videoElement.playsInline = true;
      this.videoElement.muted = true; // Muted for autoplay support

      this.videoElement.onloadedmetadata = () => {
        resolve(this.videoElement!);
      };

      this.videoElement.onerror = (e) => {
        reject(new Error(`Failed to load video: ${e}`));
      };

      if (this.objectUrl) {
        this.videoElement.src = this.objectUrl;
      }
    });
  }

  async play(): Promise<void> {
    if (this.videoElement) {
      await this.videoElement.play();
      this.isPlaying = true;
    }
  }

  pause(): void {
    if (this.videoElement) {
      this.videoElement.pause();
      this.isPlaying = false;
    }
  }

  stop(): void {
    this.pause();
    if (this.videoElement) {
      this.videoElement.currentTime = 0;
    }
    this.cleanup();
  }

  seek(time: number): void {
    if (this.videoElement) {
      this.videoElement.currentTime = Math.max(0, Math.min(time, this.videoElement.duration));
    }
  }

  getMetadata(): VideoMetadata {
    if (!this.videoElement) {
      return { width: 0, height: 0, duration: 0, fps: 30, isRealtime: false };
    }

    return {
      width: this.videoElement.videoWidth,
      height: this.videoElement.videoHeight,
      duration: this.videoElement.duration || 0,
      fps: 30, // Estimated, actual FPS may vary
      isRealtime: false,
    };
  }

  get isRealtime(): boolean {
    return false;
  }

  get isSeekable(): boolean {
    return true;
  }

  private cleanup(): void {
    if (this.objectUrl && this.objectUrl.startsWith('blob:')) {
      URL.revokeObjectURL(this.objectUrl);
    }
    this.objectUrl = null;
    this.videoElement = null;
  }

  /**
   * Create from a File object.
   */
  static fromFile(file: File): FileVideoSource {
    return new FileVideoSource(file);
  }

  /**
   * Create from a URL.
   */
  static fromUrl(url: string): FileVideoSource {
    return new FileVideoSource(url);
  }
}
