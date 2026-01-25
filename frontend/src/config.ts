/**
 * Application configuration with environment-based settings.
 */

// Backend API URL - uses environment variable in production
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// WebSocket URL - derived from API URL
export const WS_URL = (() => {
  const base = API_BASE_URL.replace(/^http/, 'ws');
  return `${base}/api/ws/pose`;
})();

// Environment flags
export const IS_PRODUCTION = import.meta.env.PROD;
export const IS_DEVELOPMENT = import.meta.env.DEV;

// MediaPipe model path
export const MEDIAPIPE_MODEL_PATH = '/models/pose_landmarker_lite.task';

export default {
  API_BASE_URL,
  WS_URL,
  IS_PRODUCTION,
  IS_DEVELOPMENT,
  MEDIAPIPE_MODEL_PATH,
};
