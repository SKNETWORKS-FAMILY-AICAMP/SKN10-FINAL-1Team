"use client"

import { Paperclip, X } from "lucide-react"

interface FilePreviewProps {
  attachedFiles: File[]
  onRemoveFile: (index: number) => void
}

export function FilePreview({ attachedFiles, onRemoveFile }: FilePreviewProps) {
  if (attachedFiles.length === 0) return null

  return (
    <div className="mb-3 flex flex-wrap gap-2">
      {attachedFiles.map((file, index) => (
        <div
          key={index}
          className="flex items-center space-x-2 bg-blue-50 text-blue-700 px-3 py-1 rounded-full text-sm"
        >
          <Paperclip className="h-3 w-3" />
          <span className="truncate max-w-32">{file.name}</span>
          <button onClick={() => onRemoveFile(index)} className="text-blue-500 hover:text-blue-700">
            <X className="h-3 w-3" />
          </button>
        </div>
      ))}
    </div>
  )
}
