"use client"

import { Button } from "@/components/ui/button"
import { MessageCircle, Trash2 } from "lucide-react"
import type { TSession } from "@/types/chat"

interface SessionItemProps {
  session: TSession
  isActive: boolean
  onSelect: () => void
  onDelete: () => void
}

export function SessionItem({ session, isActive, onSelect, onDelete }: SessionItemProps) {
  return (
    <div
      className={`group relative p-3 rounded-lg cursor-pointer transition-colors ${
        isActive ? "bg-blue-100 border border-blue-200" : "bg-gray-50 hover:bg-gray-100"
      }`}
    >
      <div onClick={onSelect} className="flex items-start space-x-3">
        <MessageCircle className="h-4 w-4 text-blue-600 mt-1 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-800 truncate">{session.title}</p>
          <p className="text-xs text-gray-400 mt-1">{new Date(session.started_at).toLocaleDateString()}</p>
        </div>
      </div>

      <Button
        variant="ghost"
        size="sm"
        onClick={(e) => {
          e.stopPropagation()
          onDelete()
        }}
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 h-6 w-6 text-gray-400 hover:text-red-500"
      >
        <Trash2 className="h-3 w-3" />
      </Button>
    </div>
  )
}
