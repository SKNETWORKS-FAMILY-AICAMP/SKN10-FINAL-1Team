"use client"

import { Button } from "@/components/ui/button"
import { Menu, LogOut } from "lucide-react"
import type { TSession } from "@/types/chat"
import useUser from '../hooks/useUser'

interface ChatHeaderProps {
  isSidebarOpen: boolean
  onToggleSidebar: () => void
  sessions: TSession[]
  activeSessionId: string | null
}

export function ChatHeader({ isSidebarOpen, onToggleSidebar, sessions, activeSessionId }: ChatHeaderProps) {
  const { user, isLoading } = useUser();

  return (
    <div className="bg-white border-b border-gray-200 p-4 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        {!isSidebarOpen && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleSidebar}
            className="text-gray-500 hover:text-gray-700 hover:bg-gray-100"
          >
            <Menu className="h-5 w-5" />
          </Button>
        )}
        <h2 className="text-lg font-semibold text-gray-800">
          {sessions.find((s) => s.id === activeSessionId)?.title || "채팅"}
        </h2>
      </div>
      <div className="flex items-center space-x-4">
        {isLoading ? (
          <span className="text-sm text-gray-500">Loading...</span>
        ) : user ? (
          <span className="text-sm font-medium text-gray-700">
            안녕하세요, {user.name || '사용자'}님
          </span>
        ) : (
          <span className="text-sm font-medium text-gray-500">
            Not logged in
          </span>
        )}
        <Button variant="outline" size="sm" className="text-red-600 hover:bg-red-50">
          <LogOut className="h-4 w-4 mr-2" />
          로그아웃
        </Button>
      </div>
    </div>
  )
}
