/**
 * Motion buffer for tracking pose landmarks over time.
 * Used for local motion analysis before sending to backend.
 */

import { PoseLandmark } from './PoseDetector';

export interface BufferedFrame {
  landmarks: PoseLandmark[];
  timestamp: number;
}

export class MotionBuffer {
  private frames: BufferedFrame[] = [];
  private maxSize: number;

  constructor(maxSize: number = 60) {
    this.maxSize = maxSize;
  }

  /**
   * Add a frame to the buffer.
   */
  addFrame(landmarks: PoseLandmark[], timestamp: number): void {
    this.frames.push({ landmarks, timestamp });

    // Remove oldest frame if buffer is full
    if (this.frames.length > this.maxSize) {
      this.frames.shift();
    }
  }

  /**
   * Get all frames in the buffer.
   */
  getFrames(): BufferedFrame[] {
    return [...this.frames];
  }

  /**
   * Get the most recent frame.
   */
  getLatestFrame(): BufferedFrame | null {
    return this.frames.length > 0 ? this.frames[this.frames.length - 1] : null;
  }

  /**
   * Get recent N frames.
   */
  getRecentFrames(count: number): BufferedFrame[] {
    return this.frames.slice(-count);
  }

  /**
   * Calculate displacement of a specific landmark over the buffer.
   */
  getLandmarkDisplacement(landmarkIndex: number): number {
    if (this.frames.length < 2) return 0;

    const positions = this.frames
      .filter((f) => f.landmarks[landmarkIndex]?.visibility > 0.5)
      .map((f) => f.landmarks[landmarkIndex]);

    if (positions.length < 2) return 0;

    let maxX = -Infinity, minX = Infinity;
    let maxY = -Infinity, minY = Infinity;

    for (const pos of positions) {
      maxX = Math.max(maxX, pos.x);
      minX = Math.min(minX, pos.x);
      maxY = Math.max(maxY, pos.y);
      minY = Math.min(minY, pos.y);
    }

    // Return 2D displacement magnitude
    const dx = maxX - minX;
    const dy = maxY - minY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * Calculate the velocity of a landmark (displacement per second).
   */
  getLandmarkVelocity(landmarkIndex: number): number {
    if (this.frames.length < 2) return 0;

    const recent = this.getRecentFrames(10);
    if (recent.length < 2) return 0;

    const first = recent[0];
    const last = recent[recent.length - 1];

    const timeDiff = (last.timestamp - first.timestamp) / 1000; // seconds
    if (timeDiff === 0) return 0;

    const firstPos = first.landmarks[landmarkIndex];
    const lastPos = last.landmarks[landmarkIndex];

    if (!firstPos || !lastPos || firstPos.visibility < 0.5 || lastPos.visibility < 0.5) {
      return 0;
    }

    const dx = lastPos.x - firstPos.x;
    const dy = lastPos.y - firstPos.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    return distance / timeDiff;
  }

  /**
   * Check if there's significant motion in the buffer.
   */
  hasSignificantMotion(threshold: number = 0.02): boolean {
    // Check key body landmarks for motion
    const keyLandmarks = [11, 12, 13, 14, 23, 24, 25, 26]; // Shoulders, elbows, hips, knees

    for (const idx of keyLandmarks) {
      if (this.getLandmarkDisplacement(idx) > threshold) {
        return true;
      }
    }

    return false;
  }

  /**
   * Get buffer duration in seconds.
   */
  getDuration(): number {
    if (this.frames.length < 2) return 0;
    return (this.frames[this.frames.length - 1].timestamp - this.frames[0].timestamp) / 1000;
  }

  /**
   * Clear the buffer.
   */
  clear(): void {
    this.frames = [];
  }

  /**
   * Get current buffer size.
   */
  get size(): number {
    return this.frames.length;
  }

  /**
   * Check if buffer is full.
   */
  get isFull(): boolean {
    return this.frames.length >= this.maxSize;
  }
}
