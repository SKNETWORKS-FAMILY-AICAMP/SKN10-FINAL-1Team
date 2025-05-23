"use client"

import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Separator } from "@/components/ui/separator"
import { CodeIcon, FileTextIcon, BarChart3Icon, HomeIcon, LogOutIcon, XIcon, PlusIcon } from "lucide-react"
import Link from "next/link"
import { useState, useEffect } from "react"
import { getChatSessions, type ChatSession, type AgentType, createChatSession } from "@/lib/api/chat-service" // Import chat service

// Helper function to format date
function formatDate(dateString: string): string {
  const date = new Date(dateString)
  const now = new Date()
  const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))

  if (diffDays === 0) return "Today"
  if (diffDays === 1) return "Yesterday"
  if (diffDays < 7) return `${diffDays} days ago`
  return date.toLocaleDateString()
}

export function ChatSidebar({ isOpen, onClose, user, onLogout, onSessionChange, activeSessionId }) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [isLoading, setIsLoading] = useState(false)

  // Fetch chat sessions from Django backend when component mounts
  useEffect(() => {
    const fetchSessions = async () => {
      setIsLoading(true)
      const fetchedSessions = await getChatSessions()
      setSessions(fetchedSessions)
      setIsLoading(false)
    }

    fetchSessions()
  }, [])

  // Create a new chat session
  const handleNewSession = async (agentType: AgentType = "code") => {
    setIsLoading(true)
    const newSession = await createChatSession(agentType)
    if (newSession) {
      setSessions((prev) => [newSession, ...prev])
      onSessionChange(newSession.id)
    }
    setIsLoading(false)
  }

  return (
    <>
      {/* Desktop Sidebar */}
      <div className="w-64 border-r bg-white dark:bg-slate-900 hidden md:flex flex-col">
        <div className="p-4 border-b flex justify-between items-center">
          <h2 className="font-semibold">Chat Sessions</h2>
          <Button variant="outline" size="sm" onClick={() => handleNewSession()} disabled={isLoading}>
            <PlusIcon className="h-4 w-4 mr-1" />
            New
          </Button>
        </div>

        <div className="flex-1 overflow-auto py-2">
          <div className="px-3 py-2">
            <h3 className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">NAVIGATION</h3>
            <div className="space-y-1">
              <Link
                href="/"
                className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800"
              >
                <HomeIcon className="h-4 w-4" />
                Home
              </Link>
            </div>
          </div>

          <Separator className="my-2" />

          <div className="px-3 py-2">
            <h3 className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">AGENTS</h3>
            <div className="space-y-1">
              <div
                className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                onClick={() => handleNewSession("code")}
              >
                <CodeIcon className="h-4 w-4 text-blue-500" />
                Code Analysis
              </div>
              <div
                className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                onClick={() => handleNewSession("rag")}
              >
                <FileTextIcon className="h-4 w-4 text-purple-500" />
                Document QA
              </div>
              <div
                className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                onClick={() => handleNewSession("analytics")}
              >
                <BarChart3Icon className="h-4 w-4 text-green-500" />
                Business Analysis
              </div>
            </div>
          </div>

          <Separator className="my-2" />

          <div className="px-3 py-2">
            <h3 className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">RECENT SESSIONS</h3>
            {isLoading ? (
              <div className="text-center py-4 text-sm text-slate-500">Loading sessions...</div>
            ) : sessions.length === 0 ? (
              <div className="text-center py-4 text-sm text-slate-500">No sessions found</div>
            ) : (
              <div className="space-y-1">
                {sessions.map((session) => (
                  <div
                    key={session.id}
                    className={`flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer ${
                      session.id === activeSessionId ? "bg-slate-100 dark:bg-slate-800" : ""
                    }`}
                    onClick={() => onSessionChange(session.id)}
                  >
                    {session.agent_type === "code" && <CodeIcon className="h-4 w-4 text-blue-500" />}
                    {session.agent_type === "rag" && <FileTextIcon className="h-4 w-4 text-purple-500" />}
                    {session.agent_type === "analytics" && <BarChart3Icon className="h-4 w-4 text-green-500" />}
                    <div className="flex-1 min-w-0">
                      <div className="truncate">
                        {session.title ||
                          `${session.agent_type.charAt(0).toUpperCase() + session.agent_type.slice(1)} Session`}
                      </div>
                      <div className="text-xs text-slate-500 dark:text-slate-400">{formatDate(session.started_at)}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="border-t p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Avatar className="h-8 w-8">
                <AvatarImage src={user?.avatar || "/placeholder.svg"} />
                <AvatarFallback>{user?.name?.charAt(0)}</AvatarFallback>
              </Avatar>
              <div>
                <div className="text-sm font-medium">{user?.name}</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">{user?.email}</div>
              </div>
            </div>
            <Button variant="ghost" size="icon" onClick={onLogout}>
              <LogOutIcon className="h-4 w-4" />
              <span className="sr-only">Log out</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Sidebar */}
      <Sheet open={isOpen} onOpenChange={onClose}>
        <SheetContent side="left" className="w-[300px] sm:w-[350px] p-0">
          <SheetHeader className="p-4 border-b">
            <SheetTitle className="flex items-center justify-between">
              <span>AI Agent Platform</span>
              <Button variant="ghost" size="icon" onClick={onClose}>
                <XIcon className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </Button>
            </SheetTitle>
          </SheetHeader>

          <div className="overflow-auto py-2">
            <div className="px-3 py-2">
              <h3 className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">NAVIGATION</h3>
              <div className="space-y-1">
                <Link
                  href="/"
                  className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  <HomeIcon className="h-4 w-4" />
                  Home
                </Link>
              </div>
            </div>

            <Separator className="my-2" />

            <div className="px-3 py-2">
              <h3 className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">AGENTS</h3>
              <div className="space-y-1">
                <div
                  className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                  onClick={() => {
                    handleNewSession("code")
                    onClose()
                  }}
                >
                  <CodeIcon className="h-4 w-4 text-blue-500" />
                  Code Analysis
                </div>
                <div
                  className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                  onClick={() => {
                    handleNewSession("rag")
                    onClose()
                  }}
                >
                  <FileTextIcon className="h-4 w-4 text-purple-500" />
                  Document QA
                </div>
                <div
                  className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                  onClick={() => {
                    handleNewSession("analytics")
                    onClose()
                  }}
                >
                  <BarChart3Icon className="h-4 w-4 text-green-500" />
                  Business Analysis
                </div>
              </div>
            </div>

            <Separator className="my-2" />

            <div className="px-3 py-2">
              <h3 className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">RECENT SESSIONS</h3>
              {isLoading ? (
                <div className="text-center py-4 text-sm text-slate-500">Loading sessions...</div>
              ) : sessions.length === 0 ? (
                <div className="text-center py-4 text-sm text-slate-500">No sessions found</div>
              ) : (
                <div className="space-y-1">
                  {sessions.map((session) => (
                    <div
                      key={session.id}
                      className={`flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer ${
                        session.id === activeSessionId ? "bg-slate-100 dark:bg-slate-800" : ""
                      }`}
                      onClick={() => {
                        onSessionChange(session.id)
                        onClose()
                      }}
                    >
                      {session.agent_type === "code" && <CodeIcon className="h-4 w-4 text-blue-500" />}
                      {session.agent_type === "rag" && <FileTextIcon className="h-4 w-4 text-purple-500" />}
                      {session.agent_type === "analytics" && <BarChart3Icon className="h-4 w-4 text-green-500" />}
                      <div className="flex-1 min-w-0">
                        <div className="truncate">
                          {session.title ||
                            `${session.agent_type.charAt(0).toUpperCase() + session.agent_type.slice(1)} Session`}
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-400">
                          {formatDate(session.started_at)}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="border-t p-4 mt-auto">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Avatar className="h-8 w-8">
                  <AvatarImage src={user?.avatar || "/placeholder.svg"} />
                  <AvatarFallback>{user?.name?.charAt(0)}</AvatarFallback>
                </Avatar>
                <div>
                  <div className="text-sm font-medium">{user?.name}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">{user?.email}</div>
                </div>
              </div>
              <Button variant="ghost" size="icon" onClick={onLogout}>
                <LogOutIcon className="h-4 w-4" />
                <span className="sr-only">Log out</span>
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
