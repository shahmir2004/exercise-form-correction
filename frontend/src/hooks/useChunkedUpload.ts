/**
 * React hook for chunked video upload with resume support.
 */

import { useState, useCallback, useRef } from 'react';

export interface UploadProgress {
  progress: number;
  uploadedChunks: number;
  totalChunks: number;
  status: 'idle' | 'initializing' | 'uploading' | 'paused' | 'completing' | 'complete' | 'error';
  error?: string;
  uploadId?: string;
  filePath?: string;
}

export interface UseChunkedUploadOptions {
  chunkSize?: number;
  maxRetries?: number;
  onProgress?: (progress: UploadProgress) => void;
  onComplete?: (result: { filename: string; file_path: string; size: number }) => void;
  onError?: (error: Error) => void;
}

export interface UseChunkedUploadReturn {
  upload: (file: File) => Promise<void>;
  pause: () => void;
  resume: (file: File) => Promise<void>;
  cancel: () => Promise<void>;
  progress: UploadProgress;
  isUploading: boolean;
}

const DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024; // 5MB

export function useChunkedUpload(options: UseChunkedUploadOptions = {}): UseChunkedUploadReturn {
  const {
    chunkSize = DEFAULT_CHUNK_SIZE,
    maxRetries = 3,
    onProgress,
    onComplete,
    onError,
  } = options;

  const [progress, setProgress] = useState<UploadProgress>({
    progress: 0,
    uploadedChunks: 0,
    totalChunks: 0,
    status: 'idle',
  });

  const abortControllerRef = useRef<AbortController | null>(null);
  const uploadIdRef = useRef<string | null>(null);
  const isPausedRef = useRef(false);

  const updateProgress = useCallback((update: Partial<UploadProgress>) => {
    setProgress((prev: UploadProgress) => {
      const newProgress = { ...prev, ...update };
      onProgress?.(newProgress);
      return newProgress;
    });
  }, [onProgress]);

  const uploadChunk = async (
    uploadId: string,
    chunk: Blob,
    chunkIndex: number,
    retries = 0
  ): Promise<boolean> => {
    try {
      const formData = new FormData();
      formData.append('chunk', chunk);

      const response = await fetch(
        `/api/upload/chunk/${uploadId}?chunk_index=${chunkIndex}`,
        {
          method: 'POST',
          body: formData,
          signal: abortControllerRef.current?.signal,
        }
      );

      if (!response.ok) {
        throw new Error(`Chunk ${chunkIndex} upload failed: ${response.statusText}`);
      }

      return true;
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        throw error; // Don't retry aborted requests
      }

      if (retries < maxRetries) {
        // Exponential backoff
        await new Promise((r) => setTimeout(r, 1000 * Math.pow(2, retries)));
        return uploadChunk(uploadId, chunk, chunkIndex, retries + 1);
      }

      throw error;
    }
  };

  const upload = useCallback(async (file: File) => {
    abortControllerRef.current = new AbortController();
    isPausedRef.current = false;

    const totalChunks = Math.ceil(file.size / chunkSize);

    updateProgress({
      status: 'initializing',
      totalChunks,
      uploadedChunks: 0,
      progress: 0,
    });

    try {
      // Initialize upload session
      const initResponse = await fetch('/api/upload/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: file.name,
          total_size: file.size,
          total_chunks: totalChunks,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!initResponse.ok) {
        throw new Error(`Failed to initialize upload: ${initResponse.statusText}`);
      }

      const { upload_id } = await initResponse.json();
      uploadIdRef.current = upload_id;

      updateProgress({
        status: 'uploading',
        uploadId: upload_id,
      });

      // Check for existing progress (resumable)
      let uploadedChunks = new Set<number>();
      try {
        const statusResponse = await fetch(`/api/upload/status/${upload_id}`);
        if (statusResponse.ok) {
          const status = await statusResponse.json();
          uploadedChunks = new Set(status.uploaded_chunks || []);
        }
      } catch {
        // Ignore - start fresh
      }

      // Upload chunks
      for (let i = 0; i < totalChunks; i++) {
        if (isPausedRef.current || abortControllerRef.current?.signal.aborted) {
          updateProgress({ status: 'paused' });
          return;
        }

        // Skip already uploaded chunks
        if (uploadedChunks.has(i)) {
          updateProgress({
            uploadedChunks: i + 1,
            progress: ((i + 1) / totalChunks) * 100,
          });
          continue;
        }

        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, file.size);
        const chunk = file.slice(start, end);

        await uploadChunk(upload_id, chunk, i);

        updateProgress({
          uploadedChunks: i + 1,
          progress: ((i + 1) / totalChunks) * 100,
        });
      }

      // Complete upload
      updateProgress({ status: 'completing' });

      const completeResponse = await fetch(`/api/upload/complete/${upload_id}`, {
        method: 'POST',
        signal: abortControllerRef.current.signal,
      });

      if (!completeResponse.ok) {
        throw new Error(`Failed to complete upload: ${completeResponse.statusText}`);
      }

      const result = await completeResponse.json();

      updateProgress({
        status: 'complete',
        progress: 100,
        filePath: result.file_path,
      });

      onComplete?.(result);
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        updateProgress({ status: 'paused' });
        return;
      }

      const err = error instanceof Error ? error : new Error('Upload failed');
      updateProgress({
        status: 'error',
        error: err.message,
      });
      onError?.(err);
    }
  }, [chunkSize, maxRetries, updateProgress, onComplete, onError]);

  const pause = useCallback(() => {
    isPausedRef.current = true;
    abortControllerRef.current?.abort();
    updateProgress({ status: 'paused' });
  }, [updateProgress]);

  const resume = useCallback(async (file: File) => {
    if (!uploadIdRef.current) {
      // No upload to resume, start fresh
      return upload(file);
    }

    isPausedRef.current = false;
    abortControllerRef.current = new AbortController();
    updateProgress({ status: 'uploading' });

    // Re-trigger upload - it will skip already uploaded chunks
    return upload(file);
  }, [upload, updateProgress]);

  const cancel = useCallback(async () => {
    pause();

    if (uploadIdRef.current) {
      try {
        await fetch(`/api/upload/${uploadIdRef.current}`, {
          method: 'DELETE',
        });
      } catch {
        // Ignore errors during cancel
      }
    }

    uploadIdRef.current = null;
    updateProgress({
      status: 'idle',
      progress: 0,
      uploadedChunks: 0,
      totalChunks: 0,
      uploadId: undefined,
    });
  }, [pause, updateProgress]);

  return {
    upload,
    pause,
    resume,
    cancel,
    progress,
    isUploading: progress.status === 'uploading' || progress.status === 'initializing' || progress.status === 'completing',
  };
}
