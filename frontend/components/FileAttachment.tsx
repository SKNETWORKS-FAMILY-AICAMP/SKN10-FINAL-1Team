"use client"

import type React from "react"

import { Button } from "@/components/ui/button"
import { Paperclip, X } from "lucide-react"

interface FileAttachmentProps {
  attachedFiles: File[]
  onFileAttach: (event: React.ChangeEvent<HTMLInputElement>) => void
  onRemoveFile: (index: number) => void
}

export function FileAttachment({ attachedFiles, onFileAttach, onRemoveFile }: FileAttachmentProps) {
  return (
    <>
      {/* 첨부된 파일 미리보기 */}
      {attachedFiles.length > 0 && (
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
      )}

      {/* 파일 첨부 버튼 */}
      <div className="relative">
        <input
          type="file"
          multiple
          accept="image/*,.pdf,.doc,.docx,.txt"
          onChange={onFileAttach}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          id="file-upload"
        />
        <Button variant="outline" size="sm" className="border-gray-300 text-gray-600 hover:bg-gray-50" asChild>
          <label htmlFor="file-upload" className="cursor-pointer">
            <Paperclip className="h-4 w-4" />
          </label>
        </Button>
      </div>
    </>
  )
}
