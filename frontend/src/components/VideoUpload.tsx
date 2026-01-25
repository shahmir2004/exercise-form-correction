/**
 * Video upload component with progress tracking.
 */

import React, { useCallback } from 'react';
import { Upload, Pause, Play, X, CheckCircle, AlertCircle } from 'lucide-react';
import { useChunkedUpload, UploadProgress } from '../hooks/useChunkedUpload';

export interface VideoUploadProps {
  onUploadComplete: (filePath: string) => void;
  className?: string;
}

const StatusIcon: React.FC<{ status: UploadProgress['status'] }> = ({ status }) => {
  switch (status) {
    case 'complete':
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    case 'error':
      return <AlertCircle className="w-5 h-5 text-red-500" />;
    default:
      return <Upload className="w-5 h-5 text-blue-500" />;
  }
};

export const VideoUpload: React.FC<VideoUploadProps> = ({
  onUploadComplete,
  className = '',
}) => {
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);

  const { upload, pause, resume, cancel, progress, isUploading } = useChunkedUpload({
    onComplete: (result) => {
      onUploadComplete(result.file_path);
    },
    onError: (error) => {
      console.error('Upload error:', error);
    },
  });

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  }, []);

  const handleUpload = useCallback(async () => {
    if (selectedFile) {
      await upload(selectedFile);
    }
  }, [selectedFile, upload]);

  const handleResume = useCallback(async () => {
    if (selectedFile) {
      await resume(selectedFile);
    }
  }, [selectedFile, resume]);

  const handleCancel = useCallback(async () => {
    await cancel();
    setSelectedFile(null);
  }, [cancel]);

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
  };

  return (
    <div className={`bg-gray-800 rounded-xl p-4 ${className}`}>
      <div className="flex items-center gap-2 mb-4">
        <StatusIcon status={progress.status} />
        <span className="text-white font-medium">Video Upload</span>
      </div>

      {!selectedFile ? (
        // File selection
        <div className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center">
          <input
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            className="hidden"
            id="video-upload-input"
          />
          <label
            htmlFor="video-upload-input"
            className="cursor-pointer flex flex-col items-center"
          >
            <Upload className="w-10 h-10 text-gray-400 mb-2" />
            <span className="text-gray-400">Click to select a video</span>
            <span className="text-sm text-gray-500 mt-1">
              MP4, MOV, AVI, WebM supported
            </span>
          </label>
        </div>
      ) : (
        // Upload progress
        <div className="space-y-3">
          {/* File info */}
          <div className="flex items-center justify-between text-sm">
            <span className="text-white truncate max-w-[200px]">
              {selectedFile.name}
            </span>
            <span className="text-gray-400">
              {formatSize(selectedFile.size)}
            </span>
          </div>

          {/* Progress bar */}
          <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`absolute left-0 top-0 h-full transition-all duration-300 ${
                progress.status === 'error'
                  ? 'bg-red-500'
                  : progress.status === 'complete'
                  ? 'bg-green-500'
                  : 'bg-blue-500'
              }`}
              style={{ width: `${progress.progress}%` }}
            />
          </div>

          {/* Progress info */}
          <div className="flex items-center justify-between text-xs text-gray-400">
            <span>
              {progress.status === 'complete'
                ? 'Upload complete!'
                : progress.status === 'error'
                ? progress.error || 'Upload failed'
                : progress.status === 'paused'
                ? 'Paused'
                : `${progress.uploadedChunks} / ${progress.totalChunks} chunks`}
            </span>
            <span>{progress.progress.toFixed(1)}%</span>
          </div>

          {/* Controls */}
          <div className="flex items-center justify-center gap-2">
            {progress.status === 'idle' && (
              <button
                onClick={handleUpload}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                Upload
              </button>
            )}

            {isUploading && (
              <button
                onClick={pause}
                className="p-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg transition-colors"
              >
                <Pause className="w-4 h-4" />
              </button>
            )}

            {progress.status === 'paused' && (
              <button
                onClick={handleResume}
                className="p-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
              >
                <Play className="w-4 h-4" />
              </button>
            )}

            {(progress.status === 'paused' || progress.status === 'error') && (
              <button
                onClick={handleCancel}
                className="p-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            )}

            {progress.status === 'complete' && (
              <button
                onClick={() => setSelectedFile(null)}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
              >
                Upload Another
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
