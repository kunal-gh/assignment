'use client';

import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { FileUp, X, FileText } from 'lucide-react';
import { useScreeningStore } from '@/store/screeningStore';

export default function FileUpload() {
  const { files, addFiles, removeFile } = useScreeningStore();

  const onDrop = useCallback(
    (acceptedFiles: File[]) => { addFiles(acceptedFiles); },
    [addFiles]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    multiple: true,
  });

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="brutalist-card p-4 flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 pb-3 border-b-4 border-black border-dashed flex-shrink-0">
        <FileUp className="w-5 h-5" />
        <h2 className="text-base font-black tracking-widest uppercase text-black">
          UPLOAD RESUMES
        </h2>
      </div>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          flex-1 border-4 border-dashed flex flex-col justify-center items-center text-center cursor-pointer
          transition-all duration-200 min-h-0
          ${isDragActive ? 'border-black bg-gray-100' : 'border-gray-300 hover:border-black hover:bg-gray-50'}
        `}
      >
        <input {...getInputProps()} />
        <motion.div
          animate={isDragActive ? { scale: 1.05 } : { scale: 1 }}
          transition={{ duration: 0.15 }}
          className="flex flex-col items-center py-4"
        >
          <FileUp className={`w-8 h-8 mb-2 ${isDragActive ? 'text-black' : 'text-gray-400'}`} />
          <p className="text-base font-black tracking-widest uppercase text-black mb-1">
            {isDragActive ? 'DROP FILES HERE' : 'DRAG & DROP OR CLICK'}
          </p>
          <p className="text-sm font-bold tracking-widest uppercase text-gray-400">
            PDF & DOCX · Max 10MB each
          </p>
        </motion.div>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-3 space-y-1 max-h-32 overflow-y-auto pr-1 flex-shrink-0"
        >
          <p className="text-sm font-black tracking-widest uppercase text-black mb-1">
            {files.length} FILE{files.length !== 1 ? 'S' : ''} SELECTED
          </p>
          <AnimatePresence>
            {files.map((file, index) => (
              <motion.div
                key={`${file.name}-${index}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                className="flex items-center justify-between p-2 border-2 border-black bg-white"
              >
                <div className="flex items-center space-x-2 flex-1 min-w-0">
                  <FileText className="w-4 h-4 flex-shrink-0 text-gray-600" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-bold truncate text-black">{file.name}</p>
                    <p className="text-sm text-gray-400">{formatSize(file.size)}</p>
                  </div>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); removeFile(index); }}
                  className="ml-2 p-1 border border-transparent hover:border-black transition-colors flex-shrink-0"
                  aria-label={`Remove ${file.name}`}
                >
                  <X className="w-3 h-3" />
                </button>
              </motion.div>
            ))}
          </AnimatePresence>
        </motion.div>
      )}
    </div>
  );
}
