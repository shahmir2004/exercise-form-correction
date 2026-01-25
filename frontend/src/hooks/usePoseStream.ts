/**
 * React hook for WebSocket connection to pose analysis backend.
 * Streams landmarks and receives form correction feedback.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { PoseLandmark } from '../pose';
import { WS_URL } from '../config';

export interface FormCorrectionResponse {
  state: 'idle' | 'scanning' | 'active';
  current_exercise: string | null;
  exercise_display: string;
  rep_count: number;
  rep_phase: string;
  is_rep_valid: boolean;
  violations: string[];
  corrections: string[];
  correction_message: string;
  joint_colors: Record<string, string>;
  confidence: number;
  timestamp: number;
}

export interface UsePoseStreamOptions {
  url?: string;
  clientId?: string;
  autoConnect?: boolean;
  onResponse?: (response: FormCorrectionResponse) => void;
  onError?: (error: Error) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

export interface UsePoseStreamReturn {
  isConnected: boolean;
  isConnecting: boolean;
  latestResponse: FormCorrectionResponse | null;
  error: Error | null;
  connect: () => void;
  disconnect: () => void;
  sendLandmarks: (landmarks: PoseLandmark[], timestamp: number) => void;
  reset: () => Promise<void>;
}

const DEFAULT_WS_URL = WS_URL;

// Generate stable client ID once
const stableClientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

export function usePoseStream(options: UsePoseStreamOptions = {}): UsePoseStreamReturn {
  const {
    url = DEFAULT_WS_URL,
    clientId: providedClientId,
    autoConnect = false,
    onResponse,
    onError,
    onConnect,
    onDisconnect,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [latestResponse, setLatestResponse] = useState<FormCorrectionResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);

  // Use refs to store stable values
  const wsRef = useRef<WebSocket | null>(null);
  const clientIdRef = useRef(providedClientId || stableClientId);
  const pendingMessagesRef = useRef<string[]>([]);
  const mountedRef = useRef(true);
  const connectingRef = useRef(false);

  // Store callbacks in refs to avoid dependency issues
  const callbacksRef = useRef({ onResponse, onError, onConnect, onDisconnect });
  
  // Update callbacks ref when they change
  useEffect(() => {
    callbacksRef.current = { onResponse, onError, onConnect, onDisconnect };
  }, [onResponse, onError, onConnect, onDisconnect]);

  const disconnect = useCallback(() => {
    connectingRef.current = false;
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connections - comprehensive check
    if (connectingRef.current) {
      console.log('Already connecting, skipping...');
      return;
    }
    
    if (wsRef.current) {
      const state = wsRef.current.readyState;
      if (state === WebSocket.CONNECTING || state === WebSocket.OPEN) {
        console.log('WebSocket already open or connecting, skipping...');
        return;
      }
    }

    connectingRef.current = true;
    setIsConnecting(true);
    setError(null);

    const wsUrl = `${url}/${clientIdRef.current}`;
    console.log('Connecting to WebSocket:', wsUrl);
    
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      if (!mountedRef.current) {
        ws.close();
        return;
      }
      
      console.log('WebSocket connected');
      connectingRef.current = false;
      setIsConnected(true);
      setIsConnecting(false);
      callbacksRef.current.onConnect?.();

      // Send any pending messages
      while (pendingMessagesRef.current.length > 0) {
        const msg = pendingMessagesRef.current.shift();
        if (msg) ws.send(msg);
      }
    };

    ws.onmessage = (event) => {
      if (!mountedRef.current) return;
      
      try {
        const response: FormCorrectionResponse = JSON.parse(event.data);
        setLatestResponse(response);
        callbacksRef.current.onResponse?.(response);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.onerror = (_event) => {
      if (!mountedRef.current) return;
      
      const err = new Error('WebSocket error');
      setError(err);
      callbacksRef.current.onError?.(err);
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      
      console.log('WebSocket disconnected');
      connectingRef.current = false;
      setIsConnected(false);
      setIsConnecting(false);
      callbacksRef.current.onDisconnect?.();
      wsRef.current = null;
    };

    wsRef.current = ws;
  }, [url]);

  const sendLandmarks = useCallback((landmarks: PoseLandmark[], timestamp: number) => {
    const message = JSON.stringify({
      landmarks: landmarks.map((lm) => ({
        x: lm.x,
        y: lm.y,
        z: lm.z,
        visibility: lm.visibility,
      })),
      timestamp,
    });

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(message);
    } else {
      // Queue message for when connection is established
      pendingMessagesRef.current.push(message);
      
      // Limit queue size
      if (pendingMessagesRef.current.length > 5) {
        pendingMessagesRef.current.shift();
      }
    }
  }, []);

  const reset = useCallback(async () => {
    try {
      const response = await fetch(`/api/reset/${clientIdRef.current}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error('Failed to reset session');
      }
      
      setLatestResponse(null);
    } catch (e) {
      const err = e instanceof Error ? e : new Error('Failed to reset');
      setError(err);
      callbacksRef.current.onError?.(err);
    }
  }, []);

  // Mount/unmount tracking
  useEffect(() => {
    mountedRef.current = true;
    
    return () => {
      mountedRef.current = false;
      // Clean up connection on unmount
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      connectingRef.current = false;
    };
  }, []);  // Empty deps - only run on mount/unmount

  // Auto-connect on mount if enabled (only once, disabled by default)
  useEffect(() => {
    if (!autoConnect) return;
    
    // Don't reconnect if already connected or connecting
    if (wsRef.current || connectingRef.current) return;
    
    // Small delay to prevent React strict mode double-connect
    const timer = setTimeout(() => {
      if (mountedRef.current && !wsRef.current && !connectingRef.current) {
        connect();
      }
    }, 200);
    
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);  // Empty deps - only auto-connect on initial mount

  return {
    isConnected,
    isConnecting,
    latestResponse,
    error,
    connect,
    disconnect,
    sendLandmarks,
    reset,
  };
}
