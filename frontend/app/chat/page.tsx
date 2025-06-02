"use client"

import { useState, useEffect, useRef, FormEvent } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { SendIcon } from "lucide-react"
import { ChatMessage as ChatMessageComponent } from "@/components/chat-message"
import { AgentSelector } from "@/components/agent-selector"
import { ChatHeader } from "@/components/chat-header"
import { Canvas } from "@/components/canvas"
import { ChatSidebar } from "@/components/chat-sidebar"
import { getCurrentUser, logoutUser, type User } from "@/lib/api/auth-service"
import { getChatMessages, sendChatMessage, updateSessionAgentType, type ChatMessage, type AgentType } from "@/lib/api/chat-service"
import { getDocumentSummary } from "@/lib/api/document-service"
import { getFileContent } from "@/lib/api/code-service"
import { generateBusinessChart } from "@/lib/api/analytics-service"
import { toast } from "@/hooks/use-toast"
import { isPreviewEnvironment } from "@/lib/api/mock-data"

export default function ChatPage() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedAgent, setSelectedAgent] = useState<AgentType | "auto">("auto")
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  // Define types for canvas content
  type CodeCanvasContent = {
    type: "code"
    language: string
    content: string
    title: string
  }

  type DocumentCanvasContent = {
    type: "document"
    data: any
    title: string
  }

  type ChartCanvasContent = {
    type: "chart"
    data: any
    title: string
  }

  type CanvasContent = CodeCanvasContent | DocumentCanvasContent | ChartCanvasContent | null
  
  const [canvasContent, setCanvasContent] = useState<CanvasContent>(null)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Check if user is logged in
  useEffect(() => {
    const checkUser = async () => {
      const currentUser = await getCurrentUser()
      if (!currentUser) {
        // If we're in preview mode and no user is found, redirect to demo
        if (isPreviewEnvironment()) {
          router.push("/demo")
          return
        }
        router.push("/login")
        return
      }
      setUser(currentUser)
    }

    checkUser()
  }, [router])

  // Load messages when session changes
  useEffect(() => {
    if (activeSessionId) {
      const loadMessages = async () => {
        setIsLoading(true)
        const fetchedMessages = await getChatMessages(activeSessionId)
        setMessages(fetchedMessages)
        setIsLoading(false)
      }

      loadMessages()
    } else {
      setMessages([])
    }
  }, [activeSessionId])

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      const container = messagesEndRef.current.parentElement
      if (container) {
        container.scrollTop = container.scrollHeight
      }
    }
  }, [messages])

  // Handle session change
  const handleSessionChange = (sessionId: string) => {
    setActiveSessionId(sessionId)
    // Clear canvas when changing sessions
    setCanvasContent(null)
  }

  // Handle agent type change
  const handleAgentChange = async (agentType: AgentType | "auto") => {
    setSelectedAgent(agentType)
    
    // Only update in database if it's not "auto" and we have an active session
    if (agentType !== "auto" && activeSessionId) {
      try {
        await updateSessionAgentType(activeSessionId, agentType as AgentType)
      } catch (error) {
        toast({
          title: "Failed to update agent type",
          description: "Please try again later.",
          variant: "destructive",
        })
      }
    }
  }

  // Handle sending a message
  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading || !activeSessionId) return

    // Add user message to UI immediately
    const userMessage: ChatMessage = {
      id: `temp-${Date.now()}`,
      session: activeSessionId,
      role: "user",
      content: input,
      created_at: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      // Send message to Django backend
      // This will create a ChatMessage record and trigger LLM processing
      const response = await sendChatMessage(activeSessionId, input)

      if (response) {
        // Replace the temporary message with the real one from the backend
        setMessages((prev) => prev.filter((msg) => msg.id !== userMessage.id).concat(response))

        // Process response based on agent type
        if (response.agentType === "code" && response.agentData) {
          // For code agent, fetch file content if needed
          if (response.agentData.fileId) {
            const codeContent = await getFileContent(response.agentData.fileId)
            if (codeContent) {
              setCanvasContent({
                type: "code",
                language: response.agentData.language || "javascript",
                content: codeContent,
                title: response.agentData.title || "Code",
              })
            }
          } else {
            // Use code directly from response
            setCanvasContent({
              type: "code",
              language: response.agentData.language || "javascript",
              content: response.agentData.code,
              title: response.agentData.title || "Generated Code",
            })
          }
        } else if (response.agentType === "rag" && response.agentData) {
          // For document agent, fetch document summary if needed
          if (response.agentData.documentId) {
            const documentSummary = await getDocumentSummary(response.agentData.documentId)
            if (documentSummary) {
              setCanvasContent({
                type: "document",
                data: documentSummary,
                title: documentSummary.title,
              })
            }
          } else if (response.agentData.summary) {
            // Use summary directly from response
            setCanvasContent({
              type: "document",
              data: response.agentData,
              title: response.agentData.title || "Document Summary",
            })
          }
        } else if (response.agentType === "analytics" && response.agentData) {
          // For analytics agent, generate chart if needed
          if (response.agentData.query) {
            const chartData = await generateBusinessChart(response.agentData.query)
            if (chartData) {
              setCanvasContent({
                type: "chart",
                data: chartData,
                title: chartData.title,
              })
            }
          } else {
            // Use chart data directly from response
            setCanvasContent({
              type: "chart",
              data: response.agentData,
              title: response.agentData.title || "Business Chart",
            })
          }
        }
      }
    } catch (error) {
      console.error("Error sending message:", error)
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive",
      })

      // Remove the temporary message if there was an error
      setMessages((prev) => prev.filter((msg) => msg.id !== userMessage.id))
    } finally {
      setIsLoading(false)
    }
  }

  // Handle logout
  const handleLogout = async () => {
    await logoutUser()
    router.push("/login")
  }

  if (!user) {
    return <div className="min-h-screen flex items-center justify-center">Loading...</div>
  }

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-950">
      {/* Chat Sessions Sidebar */}
      <ChatSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        user={user}
        onLogout={handleLogout}
        onSessionChange={handleSessionChange}
        activeSessionId={activeSessionId}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <ChatHeader user={user} onMenuClick={() => setIsSidebarOpen(!isSidebarOpen)} onLogout={handleLogout} />

        <div className="flex-1 overflow-hidden flex">
          {/* Chat Area */}
          <div className="flex-1 flex flex-col">
            {/* Messages Container */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 max-h-[calc(100vh-180px)]" style={{ scrollBehavior: "smooth" }}>
              {!activeSessionId ? (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center p-8">
                    <h2 className="text-xl font-semibold mb-2">Welcome to AI Agent Platform</h2>
                    <p className="text-slate-500 mb-4">
                      Select a chat session from the sidebar or create a new one to get started.
                    </p>
                    <Button onClick={() => setIsSidebarOpen(true)} className="md:hidden">
                      Open Sidebar
                    </Button>
                  </div>
                </div>
              ) : messages.length === 0 && !isLoading ? (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center p-8">
                    <h2 className="text-xl font-semibold mb-2">Start a conversation</h2>
                    <p className="text-slate-500">Ask a question about code, documents, or business data.</p>
                  </div>
                </div>
              ) : (
                <>
                  {messages.map((message) => (
                    <ChatMessageComponent key={message.id} message={message} />
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
                </>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Agent Selector */}
            {activeSessionId && <AgentSelector selectedAgent={selectedAgent} onAgentChange={handleAgentChange} />}

            {/* Input Form */}
            {activeSessionId && (
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
            )}
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
