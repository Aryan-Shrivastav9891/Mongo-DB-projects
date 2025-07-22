"use client";

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';

interface UploadCardProps {
  onFileSelected: (file: File) => void;
}

export function UploadCard({ onFileSelected }: UploadCardProps) {
  const [preview, setPreview] = useState<string | null>(null);
  
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) {
      toast.error('Please upload a valid image file (JPG/PNG)');
      return;
    }
    
    const file = acceptedFiles[0];
    if (!file.type.match(/image\/(jpeg|png|jpg)/)) {
      toast.error('Only JPG and PNG images are supported');
      return;
    }

    // Create a preview
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result as string);
      onFileSelected(file);
    };
    reader.readAsDataURL(file);
    
    toast.success('Chart uploaded successfully!');
  }, [onFileSelected]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': [],
      'image/png': [],
    },
    maxFiles: 1,
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="w-full border-purple-200 bg-gradient-to-br from-white to-purple-50 shadow-md">
        <CardHeader className="border-b border-purple-100">
          <CardTitle className="text-center text-purple-800">Upload Candlestick Chart</CardTitle>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive ? 'border-purple-600 bg-purple-100/50' : 'border-purple-300 hover:border-purple-600'
            }`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <div className="space-y-4">
                <img src={preview} alt="Uploaded chart" className="max-h-[250px] mx-auto" />
                <p className="text-sm text-gray-500">Drag & drop a new file to replace</p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="mx-auto w-12 h-12 rounded-full bg-purple-100 flex items-center justify-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="text-purple-800"
                  >
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </div>
                <div>
                  <p className="font-medium text-purple-900">
                    {isDragActive
                      ? 'Drop the file here...'
                      : 'Drag & drop your candlestick chart image here'}
                  </p>
                  <p className="text-sm text-purple-600 mt-1">or click to browse files</p>
                  <p className="text-xs text-purple-500 mt-2">Supports: JPG, PNG</p>
                </div>
              </div>
            )}
          </div>
          
          {preview && (
            <div className="mt-4">
              <Button 
                onClick={() => {
                  setPreview(null);
                }}
                variant="outline" 
                className="w-full border-purple-300 text-purple-700 hover:bg-purple-50 hover:text-purple-900"
              >
                Clear Selection
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
