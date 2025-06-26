"use client"

import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Home, Settings, Plus, ChevronLeft } from "lucide-react"
import type { TSession } from "@/types/chat"
import { SessionItem } from "./SessionItem"

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  sessions: TSession[]
  activeSessionId: string | null
  onSessionSelect: (sessionId: string) => void
  onNewSession: () => void
  onDeleteSession: (sessionId: string) => void
}

export function Sidebar({
  isOpen,
  onClose,
  sessions,
  activeSessionId,
  onSessionSelect,
  onNewSession,
  onDeleteSession,
}: SidebarProps) {
  return (
    <div
      className={`
        ${isOpen ? "translate-x-0" : "-translate-x-full"}
        fixed lg:fixed z-50
        w-80 bg-white border-r border-gray-200 flex flex-col
        transition-transform duration-300 ease-in-out
        h-full
      `}
    >
      {/* 사이드바 헤더 */}
      <div className="p-4 border-b border-gray-200 flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-800">AI 챗봇</h1>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 hover:bg-gray-100"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      </div>

      {/* 네비게이션 메뉴 */}
      <div className="p-4 space-y-2">
        <a href="/">
          <Button variant="ghost" className="w-full justify-start text-gray-700 hover:bg-blue-50 hover:text-blue-600">
            <Home className="mr-3 h-4 w-4" />홈
          </Button>
        </a>

        <a href="/accounts/settings/">
          <Button variant="ghost" className="w-full justify-start text-gray-700 hover:bg-blue-50 hover:text-blue-600">
            <Settings className="mr-3 h-4 w-4" />
            설정
          </Button>
        </a>

        <Button onClick={onNewSession} className="w-full justify-start bg-blue-600 text-white hover:bg-blue-700">
          <Plus className="mr-3 h-4 w-4" />새 세션 추가
        </Button>
      </div>

      <Separator />

      {/* 세션 목록 */}
      <div className="flex-1 p-4">
        <h3 className="text-sm font-medium text-gray-500 mb-3">세션 목록</h3>
        <ScrollArea className="h-full">
          <div className="space-y-2">
            {sessions.map((session) => (
              <SessionItem
                key={session.id}
                session={session}
                isActive={activeSessionId === session.id}
                onSelect={() => onSessionSelect(session.id)}
                onDelete={() => onDeleteSession(session.id)}
              />
            ))}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}
