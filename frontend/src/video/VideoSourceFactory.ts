/**
 * Factory for creating video sources based on environment/configuration.
 */

import { VideoSource } from './VideoSource';
import { FileVideoSource } from './FileVideoSource';
import { WebcamSource, WebcamConfig } from './WebcamSource';

export type VideoSourceType = 'file' | 'webcam';

export interface VideoSourceFactoryConfig {
  type: VideoSourceType;
  file?: File;
  url?: string;
  webcamConfig?: WebcamConfig;
}

export class VideoSourceFactory {
  /**
   * Create a video source based on configuration.
   */
  static create(config: VideoSourceFactoryConfig): VideoSource {
    switch (config.type) {
      case 'file':
        if (config.file) {
          return FileVideoSource.fromFile(config.file);
        } else if (config.url) {
          return FileVideoSource.fromUrl(config.url);
        }
        throw new Error('File source requires either a File object or URL');

      case 'webcam':
        return new WebcamSource(config.webcamConfig);

      default:
        throw new Error(`Unknown video source type: ${config.type}`);
    }
  }

  /**
   * Create file source from File object.
   */
  static fromFile(file: File): VideoSource {
    return this.create({ type: 'file', file });
  }

  /**
   * Create file source from URL.
   */
  static fromUrl(url: string): VideoSource {
    return this.create({ type: 'file', url });
  }

  /**
   * Create webcam source.
   */
  static fromWebcam(config?: WebcamConfig): VideoSource {
    return this.create({ type: 'webcam', webcamConfig: config });
  }

  /**
   * Create source based on environment.
   * In development: defaults to file upload
   * In production: can default to webcam
   */
  static createForEnvironment(): VideoSource | null {
    const env = import.meta.env.MODE;
    
    if (env === 'production') {
      // In production, we might default to webcam
      // But for safety, return null and let user choose
      return null;
    }
    
    // In development, return null and let user upload a file
    return null;
  }
}
