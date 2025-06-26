"use client"

import type React from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Send, Paperclip } from "lucide-react"
import { FilePreview } from "./FilePreview"

interface MessageInputProps {
  input: string
  onInputChange: (value: string) => void
  onSubmit: (e: React.FormEvent) => void
  isLoading: boolean
  isDisabled: boolean
  attachedFiles: File[]
  onFileAttach: (event: React.ChangeEvent<HTMLInputElement>) => void
  onRemoveFile: (index: number) => void
}

export function MessageInput({
  input,
  onInputChange,
  onSubmit,
  isLoading,
  isDisabled,
  attachedFiles,
  onFileAttach,
  onRemoveFile,
}: MessageInputProps) {
  return (
    <div className="bg-white border-t border-gray-200 p-4">
      <div className="max-w-4xl mx-auto">
        <FilePreview attachedFiles={attachedFiles} onRemoveFile={onRemoveFile} />

        <form onSubmit={onSubmit} className="flex space-x-3">
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

          <Input
            value={input}
            onChange={(e) => onInputChange(e.target.value)}
            placeholder="업무 내용을 입력하세요..."
            className="flex-1 border-gray-300 focus:border-blue-500 focus:ring-blue-500"
            disabled={isDisabled || isLoading}
          />
          <Button
            type="submit"
            className="bg-blue-600 hover:bg-blue-700 text-white"
            disabled={isDisabled || isLoading || !input.trim()}
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  )
}
