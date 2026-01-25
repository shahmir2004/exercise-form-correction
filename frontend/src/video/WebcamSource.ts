/**
 * Webcam-based video source for real-time capture.
 * Used for production real-time exercise detection.
 */

import { VideoSource, VideoMetadata } from './VideoSource';

export interface WebcamConfig {
  deviceId?: string;
  width?: number;
  height?: number;
  facingMode?: 'user' | 'environment';
  frameRate?: number;
}

export class WebcamSource extends VideoSource {
  private stream: MediaStream | null = null;
  private webcamConfig: WebcamConfig;

  constructor(config: WebcamConfig = {}) {
    super({
      type: 'webcam',
      fps: config.frameRate || 30,
      width: config.width || 1280,
      height: config.height || 720,
    });

    this.webcamConfig = {
      width: 1280,
      height: 720,
      facingMode: 'user',
      frameRate: 30,
      ...config,
    };
  }

  async initialize(): Promise<HTMLVideoElement> {
    // Request camera access
    const constraints: MediaStreamConstraints = {
      video: {
        deviceId: this.webcamConfig.deviceId ? { exact: this.webcamConfig.deviceId } : undefined,
        width: { ideal: this.webcamConfig.width },
        height: { ideal: this.webcamConfig.height },
        facingMode: this.webcamConfig.facingMode,
        frameRate: { ideal: this.webcamConfig.frameRate },
      },
      audio: false,
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      throw new Error(`Failed to access webcam: ${error}`);
    }

    return new Promise((resolve, reject) => {
      this.videoElement = document.createElement('video');
      this.videoElement.playsInline = true;
      this.videoElement.muted = true;
      this.videoElement.srcObject = this.stream;

      this.videoElement.onloadedmetadata = () => {
        resolve(this.videoElement!);
      };

      this.videoElement.onerror = (e) => {
        reject(new Error(`Failed to initialize webcam video: ${e}`));
      };
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
    this.cleanup();
  }

  seek(_time: number): void {
    // Webcam doesn't support seeking
    console.warn('WebcamSource does not support seeking');
  }

  getMetadata(): VideoMetadata {
    if (!this.videoElement) {
      return { width: 0, height: 0, duration: Infinity, fps: 30, isRealtime: true };
    }

    return {
      width: this.videoElement.videoWidth,
      height: this.videoElement.videoHeight,
      duration: Infinity, // Live stream has no duration
      fps: this.webcamConfig.frameRate || 30,
      isRealtime: true,
    };
  }

  get isRealtime(): boolean {
    return true;
  }

  get isSeekable(): boolean {
    return false;
  }

  private cleanup(): void {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.videoElement) {
      this.videoElement.srcObject = null;
      this.videoElement = null;
    }
  }

  /**
   * Get list of available video input devices.
   */
  static async getAvailableDevices(): Promise<MediaDeviceInfo[]> {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === 'videoinput');
  }

  /**
   * Check if webcam access is available.
   */
  static async isAvailable(): Promise<boolean> {
    try {
      const devices = await this.getAvailableDevices();
      return devices.length > 0;
    } catch {
      return false;
    }
  }
}
