"use client"

import { useState, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { SendIcon } from "lucide-react"
import { ChatMessage } from "@/components/chat-message"
import { AgentSelector } from "@/components/agent-selector"
import { mockAgentResponse } from "@/lib/mock-agent"
import { ChatHeader } from "@/components/chat-header"
import { Canvas } from "@/components/canvas"

export default function ChatPage() {
  const router = useRouter()
  const [user, setUser] = useState(null)
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedAgent, setSelectedAgent] = useState("auto")
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [canvasContent, setCanvasContent] = useState(null)
  const [chatSessions, setChatSessions] = useState([
    { id: "session1", name: "Code Analysis Session", lastMessage: "Yesterday", active: true },
    { id: "session2", name: "Document Research", lastMessage: "2 days ago", active: false },
    { id: "session3", name: "Business Metrics Review", lastMessage: "Last week", active: false },
  ])
  const messagesEndRef = useRef(null)

  useEffect(() => {
    // Check if user is logged in
    const userData = localStorage.getItem("user")
    if (!userData) {
      router.push("/login")
      return
    }

    setUser(JSON.parse(userData))

    // Add welcome message
    setMessages([
      {
        id: "welcome",
        role: "system",
        content: "Welcome to the AI Agent Platform! How can I assist you today?",
        timestamp: new Date().toISOString(),
      },
    ])
  }, [router])

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    // Simulate agent processing
    setTimeout(() => {
      const response = mockAgentResponse(input, selectedAgent)
      setMessages((prev) => [...prev, response])
      setIsLoading(false)

      // If the response has code or other structured data, show it in the canvas
      if (response.agentType === "code" && response.agentData) {
        setCanvasContent({
          type: "code",
          language: response.agentData.language,
          content: response.agentData.code,
          title: "Generated Code",
        })
      } else if (response.agentType === "business" && response.agentData) {
        setCanvasContent({
          type: "chart",
          data: response.agentData,
          title: response.agentData.title,
        })
      } else if (response.agentType === "document" && response.agentData) {
        setCanvasContent({
          type: "document",
          data: response.agentData,
          title: response.agentData.title,
        })
      }
    }, 1500)
  }

  const handleLogout = () => {
    localStorage.removeItem("user")
    router.push("/login")
  }

  const handleSessionChange = (sessionId) => {
    // In a real app, this would load the selected chat session
    setChatSessions((prev) =>
      prev.map((session) => ({
        ...session,
        active: session.id === sessionId,
      })),
    )

    // For demo purposes, just clear the messages and add a welcome message
    setMessages([
      {
        id: "welcome-new",
        role: "system",
        content: `Switched to ${chatSessions.find((s) => s.id === sessionId)?.name}. How can I assist you?`,
        timestamp: new Date().toISOString(),
      },
    ])

    // Clear canvas
    setCanvasContent(null)
  }

  const createNewSession = () => {
    const newSession = {
      id: `session${chatSessions.length + 1}`,
      name: `New Chat ${chatSessions.length + 1}`,
      lastMessage: "Just now",
      active: true,
    }

    setChatSessions((prev) =>
      prev
        .map((session) => ({
          ...session,
          active: false,
        }))
        .concat(newSession),
    )

    // Clear messages and canvas
    setMessages([
      {
        id: "welcome-new",
        role: "system",
        content: "Started a new chat session. How can I assist you?",
        timestamp: new Date().toISOString(),
      },
    ])
    setCanvasContent(null)
  }

  if (!user) {
    return <div className="min-h-screen flex items-center justify-center">Loading...</div>
  }

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-950">
      {/* Chat Sessions Sidebar */}
      <div
        className={`border-r bg-white dark:bg-slate-900 w-64 flex-shrink-0 ${isSidebarOpen ? "block" : "hidden"} md:block`}
      >
        <div className="p-4 border-b flex justify-between items-center">
          <h2 className="font-semibold">Chat Sessions</h2>
          <Button variant="outline" size="sm" onClick={createNewSession}>
            New
          </Button>
        </div>
        <div className="overflow-auto h-[calc(100%-60px)]">
          {chatSessions.map((session) => (
            <div
              key={session.id}
              className={`p-3 border-b cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-800 ${
                session.active ? "bg-slate-100 dark:bg-slate-800" : ""
              }`}
              onClick={() => handleSessionChange(session.id)}
            >
              <div className="font-medium truncate">{session.name}</div>
              <div className="text-xs text-slate-500 dark:text-slate-400 truncate">{session.lastMessage}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <ChatHeader user={user} onMenuClick={() => setIsSidebarOpen(!isSidebarOpen)} onLogout={handleLogout} />

        <div className="flex-1 overflow-hidden flex">
          {/* Chat Area */}
          <div className="flex-1 flex flex-col">
            {/* Messages Container */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {isLoading && (
                <div className="flex items-start gap-3">
                  <Avatar className="h-8 w-8">
                    <AvatarImage src="/placeholder.svg?height=32&width=32" />
                    <AvatarFallback>AI</AvatarFallback>
                  </Avatar>
                  <div className="flex space-x-2 mt-2">
                    <div className="h-3 w-3 bg-slate-300 dark:bg-slate-600 rounded-full animate-bounce"></div>
                    <div
                      className="h-3 w-3 bg-slate-300 dark:bg-slate-600 rounded-full animate-bounce"
                      style={{ animationDelay: "0.2s" }}
                    ></div>
                    <div
                      className="h-3 w-3 bg-slate-300 dark:bg-slate-600 rounded-full animate-bounce"
                      style={{ animationDelay: "0.4s" }}
                    ></div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Agent Selector */}
            <AgentSelector selectedAgent={selectedAgent} onAgentChange={setSelectedAgent} />

            {/* Input Form */}
            <div className="border-t p-4">
              <form onSubmit={handleSendMessage} className="flex gap-2">
                <Input
                  placeholder="Ask a question about code, documents, or business data..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  disabled={isLoading}
                  className="flex-1"
                />
                <Button type="submit" disabled={isLoading || !input.trim()}>
                  <SendIcon className="h-5 w-5" />
                  <span className="sr-only">Send</span>
                </Button>
              </form>
            </div>
          </div>

          {/* Canvas Area */}
          {canvasContent ? (
            <div className="w-1/2 border-l bg-white dark:bg-slate-900 overflow-auto hidden md:block">
              <Canvas content={canvasContent} onClose={() => setCanvasContent(null)} />
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
