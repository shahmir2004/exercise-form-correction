/**
 * Skeleton overlay component that draws pose landmarks on a canvas.
 * Color-codes joints based on form correction feedback.
 */

import React, { useRef, useEffect, useCallback } from 'react';
import { PoseLandmark, getPoseConnections, getLandmarkName } from '../pose';

export interface SkeletonOverlayProps {
  landmarks: PoseLandmark[] | null;
  jointColors?: Record<string, string>;
  width: number;
  height: number;
  showLabels?: boolean;
  lineWidth?: number;
  pointRadius?: number;
}

// Color palette
const COLORS = {
  valid: '#22c55e',      // Green
  invalid: '#ef4444',    // Red
  warning: '#eab308',    // Yellow
  neutral: '#3b82f6',    // Blue
  connection: '#94a3b8', // Gray for connections
};

// Joint groups for different body parts
const JOINT_GROUPS = {
  face: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  torso: [11, 12, 23, 24],
  leftArm: [11, 13, 15, 17, 19, 21],
  rightArm: [12, 14, 16, 18, 20, 22],
  leftLeg: [23, 25, 27, 29, 31],
  rightLeg: [24, 26, 28, 30, 32],
};

export const SkeletonOverlay: React.FC<SkeletonOverlayProps> = ({
  landmarks,
  jointColors = {},
  width,
  height,
  showLabels = false,
  lineWidth = 3,
  pointRadius = 6,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const getJointColor = useCallback((index: number): string => {
    const name = getLandmarkName(index);
    const colors = jointColors as Record<string, string>;
    
    // Check if there's a specific color for this joint
    if (colors[name]) {
      const color = colors[name].toLowerCase();
      if (color === 'green' || color === 'valid') return COLORS.valid;
      if (color === 'red' || color === 'invalid') return COLORS.invalid;
      if (color === 'yellow' || color === 'warning') return COLORS.warning;
      return color; // Use custom color directly
    }
    
    return COLORS.neutral;
  }, [jointColors]);

  const getConnectionColor = useCallback((startIdx: number, endIdx: number): string => {
    const startColor = getJointColor(startIdx);
    const endColor = getJointColor(endIdx);
    
    // If either end is invalid, connection is invalid
    if (startColor === COLORS.invalid || endColor === COLORS.invalid) {
      return COLORS.invalid;
    }
    if (startColor === COLORS.warning || endColor === COLORS.warning) {
      return COLORS.warning;
    }
    if (startColor === COLORS.valid && endColor === COLORS.valid) {
      return COLORS.valid;
    }
    
    return COLORS.connection;
  }, [getJointColor]);

  const drawSkeleton = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !landmarks) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Get pose connections
    const connections = getPoseConnections();

    // Draw connections first (so points appear on top)
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const [startIdx, endIdx] of connections) {
      const start = landmarks[startIdx];
      const end = landmarks[endIdx];

      // Skip if either point is not visible
      if (!start || !end || start.visibility < 0.5 || end.visibility < 0.5) {
        continue;
      }

      const startX = start.x * width;
      const startY = start.y * height;
      const endX = end.x * width;
      const endY = end.y * height;

      const color = getConnectionColor(startIdx, endIdx);

      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }

    // Draw points
    for (let i = 0; i < landmarks.length; i++) {
      const landmark = landmarks[i];
      
      // Skip if not visible
      if (landmark.visibility < 0.5) continue;

      // Skip face landmarks except nose for cleaner visualization
      if (JOINT_GROUPS.face.includes(i) && i !== 0) continue;

      const x = landmark.x * width;
      const y = landmark.y * height;
      const color = getJointColor(i);

      // Draw outer circle (border)
      ctx.beginPath();
      ctx.arc(x, y, pointRadius + 2, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();

      // Draw inner circle (colored)
      ctx.beginPath();
      ctx.arc(x, y, pointRadius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // Draw label if enabled
      if (showLabels) {
        const label = getLandmarkName(i).replace(/_/g, ' ');
        ctx.font = '10px Arial';
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.strokeText(label, x + pointRadius + 4, y + 4);
        ctx.fillText(label, x + pointRadius + 4, y + 4);
      }
    }
  }, [landmarks, width, height, jointColors, lineWidth, pointRadius, showLabels, getJointColor, getConnectionColor]);

  useEffect(() => {
    drawSkeleton();
  }, [drawSkeleton]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="skeleton-overlay"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    />
  );
};
