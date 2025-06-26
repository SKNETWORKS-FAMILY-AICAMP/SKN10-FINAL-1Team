"use client"

import { Button } from "@/components/ui/button"
import { AlertCircle } from "lucide-react"

interface DeleteConfirmDialogProps {
  isOpen: boolean
  onConfirm: () => void
  onCancel: () => void
}

export function DeleteConfirmDialog({ isOpen, onConfirm, onCancel }: DeleteConfirmDialogProps) {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-sm mx-4">
        <div className="flex items-center space-x-3 mb-4">
          <AlertCircle className="h-5 w-5 text-red-500" />
          <h3 className="text-lg font-semibold text-gray-800">세션 삭제</h3>
        </div>
        <p className="text-gray-600 mb-6">이 세션을 삭제하시겠습니까? 삭제된 세션은 복구할 수 없습니다.</p>
        <div className="flex space-x-3 justify-end">
          <Button variant="outline" onClick={onCancel} className="text-gray-600">
            취소
          </Button>
          <Button onClick={onConfirm} className="bg-red-600 hover:bg-red-700 text-white">
            삭제
          </Button>
        </div>
      </div>
    </div>
  )
}
