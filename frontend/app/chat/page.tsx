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
import { type ChatMessage, type AgentType } from "@/lib/api/chat-service"
import { getDocumentSummary } from "@/lib/api/document-service"
import { getFileContent } from "@/lib/api/code-service"
import { generateBusinessChart } from "@/lib/api/analytics-service"
import { toast } from "@/hooks/use-toast"
import { isPreviewEnvironment } from "@/lib/api/mock-data"
import { createChatWebSocket } from "@/lib/api/fastapi-service"
import { 
  getUserChatSessions, 
  createChatSessionInDB, 
  getChatMessagesFromDB, 
  saveChatMessageToDB,
  deleteChatSessionFromDB,
  updateChatSessionTitleInDB,
  updateSessionAgentTypeInDB
} from "@/lib/api/postgres-chat-service"

export default function ChatPage() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isReceivingResponse, setIsReceivingResponse] = useState(false) // Track when first token arrives
  const [selectedAgent, setSelectedAgent] = useState<AgentType | "auto">("auto")
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  
  // Edit session title states
  const [isEditingTitle, setIsEditingTitle] = useState(false)
  const [editTitle, setEditTitle] = useState("")
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
  const [sessions, setSessions] = useState<any[]>([])
  const [activeSession, setActiveSession] = useState<any | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Check if user is logged in
  useEffect(() => {
    const checkUser = async () => {
      const currentUser = await getCurrentUser()
      if (!currentUser) {
        // If we're in preview mode and no user is found, redirect to demo
        if (isPreviewEnvironment()) {
          router.push("/demo")
          return null
        }
        router.push("/login")
        return null
      }
      setUser(currentUser)
      return currentUser
    }

    const initializeUserData = async (user: User) => {
      if (user && user.id) {
        // Fetch user sessions after login
        await fetchUserSessions(user.id)
      }
    }
    
    checkUser().then((userData) => {
      if (userData) {
        initializeUserData(userData)
      }
    })
  }, [router])

  // Load messages when session changes
  useEffect(() => {
    if (activeSessionId) {
      const loadMessages = async () => {
        setIsLoading(true)
        // Use direct PostgreSQL access to get messages
        const fetchedMessages = await getChatMessagesFromDB(activeSessionId)
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

  // Fetch user sessions
  const fetchUserSessions = async (userId: string): Promise<any[]> => {
    try {
      const sessions = await getUserChatSessions(userId)
      setSessions(sessions)
      
      // Get first active session or null if none exists
      const activeSession = sessions.find(s => !s.ended_at) || sessions[0] || null
      setActiveSession(activeSession)
      
      if (activeSession) {
        // Set title for editing to current title
        setEditTitle(activeSession.title || '')
        // Set active session ID
        setActiveSessionId(activeSession.id)
        // Load messages for active session
        fetchSessionMessages(activeSession.id)
      }
      
      // Return the sessions for further use
      return sessions
    } catch (error) {
      console.error("Error fetching sessions:", error)
      toast({
        title: "Error",
        description: "Failed to load chat sessions",
        variant: "destructive",
      })
      
      // Return empty array in case of error
      return []
    }
  }
  
  // Fetch messages for a session
  const fetchSessionMessages = async (sessionId: string) => {
    try {
      const messages = await getChatMessagesFromDB(sessionId)
      setMessages(messages)
    } catch (error) {
      console.error("Error fetching messages:", error)
      toast({
        title: "Error",
        description: "Failed to load messages",
        variant: "destructive",
      })
    }
  }

  // Handle session change
  const handleSessionChange = (sessionId: string) => {
    // Find the selected session
    const selected = sessions.find(s => s.id === sessionId)
    if (selected) {
      setActiveSession(selected)
      setActiveSessionId(sessionId)
      setEditTitle(selected.title || '')
      // Load messages for the selected session
      fetchSessionMessages(sessionId)
      // Clear canvas when changing sessions
      setCanvasContent(null)
    }
  }
  
  // Start editing session title
  const handleEditTitle = () => {
    if (activeSession) {
      setEditTitle(activeSession.title || '')
      setIsEditingTitle(true)
    }
  }
  
  // Save edited session title
  const handleSaveTitle = async () => {
    if (!activeSession || !editTitle.trim()) return
    
    try {
      const updatedSession = await updateChatSessionTitleInDB(activeSession.id, editTitle.trim())
      if (updatedSession) {
        // Update active session
        setActiveSession(updatedSession)
        
        // Refresh sessions list from database
        if (user && user.id) {
          await fetchUserSessions(user.id)
        }
      }
      setIsEditingTitle(false)
    } catch (error) {
      console.error("Error updating session title:", error)
      toast({
        title: "Error",
        description: "Failed to update session title",
        variant: "destructive",
      })
    }
  }
  
  // Cancel title editing
  const handleCancelEditTitle = () => {
    setIsEditingTitle(false)
    if (activeSession) {
      setEditTitle(activeSession.title || '')
    }
  }
  
  // Delete session
  const handleDeleteSession = async (sessionId: string) => {
    if (!confirm("이 세션을 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.")) {
      return
    }
    
    try {
      const success = await deleteChatSessionFromDB(sessionId)
      if (success) {
        // If active session was deleted, clear the current session
        if (activeSession && activeSession.id === sessionId) {
          setActiveSession(null)
          setActiveSessionId(null)
          setMessages([])
        }
        
        // Refresh sessions list from database
        if (user && user.id) {
          const newSessions = await fetchUserSessions(user.id)
          
          // If active session was deleted, select a new session if available
          if (activeSession && activeSession.id === sessionId && newSessions.length > 0) {
            const newActiveSession = newSessions[0]
            setActiveSession(newActiveSession)
            setActiveSessionId(newActiveSession.id)
            fetchSessionMessages(newActiveSession.id)
            setEditTitle(newActiveSession.title || '')
          }
        }
        
        toast({
          title: "삭제 성공",
          description: "채팅 세션이 성공적으로 삭제되었습니다",
        })
      }
    } catch (error) {
      console.error("Error deleting session:", error)
      toast({
        title: "Error",
        description: "Failed to delete chat session",
        variant: "destructive",
      })
    }
  }

  // Handle agent type change
  const handleAgentChange = async (agentType: AgentType | "auto") => {
    setSelectedAgent(agentType)
    
    // Only update in database if it's not "auto" and we have an active session
    if (agentType !== "auto" && activeSessionId) {
      try {
        await updateSessionAgentTypeInDB(activeSessionId, agentType as AgentType)
      } catch (error) {
        toast({
          title: "Error",
          description: "Failed to update agent type. Please try again.",
          variant: "destructive",
        })
      }
    }
  }
  
  // Create a new chat session
  const handleNewSession = async (agentType: AgentType = "code") => {
    setIsLoading(true)
    try {
      const newSession = await createChatSessionInDB(user?.id || '', agentType)
      if (newSession) {
        // 새 세션 생성 후 세션 목록을 갱신하고 새 세션을 활성화
        await fetchUserSessions(user?.id || '')
        handleSessionChange(newSession.id)
        toast({
          title: "새 세션 생성",
          description: "새 채팅 세션이 생성되었습니다.",
        })
      }
    } catch (error) {
      console.error("Error creating new session:", error)
      toast({
        title: "Error",
        description: "Failed to create new chat session",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  // Handle sending a message
  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault()

    if (!input.trim() || isLoading || !activeSessionId) return

    const userInput = input.trim()
  
    // Add user message to UI immediately
    const userMessage: ChatMessage = {
      id: `temp-${Date.now()}`,
      session: activeSessionId,
      role: "user",
      content: userInput,
      created_at: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)
    setIsReceivingResponse(false) // Reset the response receiving flag

    try {
      // 1. Save user message directly to PostgreSQL
      const savedUserMessage = await saveChatMessageToDB(activeSessionId, "user", userInput)
      
      // Create placeholder for AI response
      const placeholderResponse: ChatMessage = {
        id: `assistant-placeholder-${Date.now()}`,
        session: activeSessionId,
        role: "assistant",
        content: "", // Empty content that will be filled via streaming
        created_at: new Date().toISOString(),
      }
      
      // Add the placeholder for streaming
      // When AI message box is created, set isReceivingResponse to true to hide loading indicator
      setIsReceivingResponse(true)
      
      setMessages((prev) => {
        // Replace the temp user message with the saved one if available
        const filtered = prev.filter((msg) => msg.id !== userMessage.id)
        return [...filtered, savedUserMessage || userMessage, placeholderResponse]
      })
      
      // Track current content for streaming updates
      let streamedContent = ""
      let currentAgentType: AgentType | null = null
      const aiMessageId = placeholderResponse.id
      
      // 3. Create WebSocket connection for streaming
      const wsConnection = createChatWebSocket(
        activeSessionId, // Use session ID as thread ID for consistency
        // On token event - update the placeholder message with streamed tokens
        (token) => {
          // Mark that we're receiving a response when the first token arrives
          setIsReceivingResponse(true)
          
          // Update the streaming content
          streamedContent += token
          
          // Update the AI response as tokens stream in
          setMessages((prev) => {
            return prev.map((msg) => {
              if (msg.id === placeholderResponse.id) {
                return { ...msg, content: streamedContent }
              }
              return msg
            })
          })
        },
        // On agent change event
        (agent) => {
          if (agent === "code" || agent === "rag" || agent === "analytics") {
            currentAgentType = agent
          }
        },
        // On tool start event
        (toolData) => {
          console.log("Tool execution started:", toolData)
          // Could show a UI indicator that a tool is running
        },
        // On tool end event - update canvas with data based on agent type
        (resultData) => {
          console.log("Tool execution ended:", resultData)
          
          // Process canvas content based on agent type and tool result
          if (currentAgentType === "code" && resultData.code) {
            setCanvasContent({
              type: "code",
              language: resultData.language || "javascript",
              content: resultData.code,
              title: resultData.title || "Generated Code",
            })
          } else if (currentAgentType === "rag" && resultData.document) {
            setCanvasContent({
              type: "document",
              data: {
                content: resultData.document,
                title: resultData.title || "Document",
              },
              title: resultData.title || "Document Summary",
            })
          } else if (currentAgentType === "analytics" && resultData.chart) {
            setCanvasContent({
              type: "chart",
              data: resultData.chart,
              title: resultData.title || "Business Chart",
            })
          }
        },
        // On error event
        (error) => {
          console.error("Streaming error:", error)
          toast({
            title: "Streaming Error",
            description: error || "An error occurred during message streaming",
            variant: "destructive",
          })
        },
        // On done event - finalize the message and save to database
        async () => {
          // Save the final assistant message to PostgreSQL
          if (streamedContent) {
            const savedAssistantMessage = await saveChatMessageToDB(
              activeSessionId,
              "assistant", 
              streamedContent
            )
            
            // Update UI with the saved message (includes proper ID from database)
            if (savedAssistantMessage) {
              setMessages((prev) => {
                return prev.map((msg) => {
                  if (msg.id === placeholderResponse.id) {
                    return savedAssistantMessage
                  }
                  return msg
                })
              })
            }
          }
          
          setIsLoading(false)
        }
      )
      
      // 4. Send the message over WebSocket
      if (wsConnection) {
        wsConnection.sendMessage(userInput)
      } else {
        throw new Error("Failed to establish WebSocket connection")
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
      setIsLoading(false)
    }
  }

// ...
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
        sessions={sessions}
        isLoading={isLoading}
        onNewSession={handleNewSession}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <ChatHeader 
          user={user} 
          onMenuClick={() => setIsSidebarOpen(!isSidebarOpen)} 
          onLogout={handleLogout} 
          activeSession={activeSession}
          isEditingTitle={isEditingTitle}
          editTitle={editTitle}
          onEditTitle={handleEditTitle}
          onSaveTitle={handleSaveTitle}
          onCancelEditTitle={handleCancelEditTitle}
          onDeleteSession={handleDeleteSession}
          onEditTitleChange={(e) => setEditTitle(e.target.value)}
        />

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
                  {messages.map((message, index) => (
                    <ChatMessageComponent
                      key={message.id}
                      message={message}
                      isStreaming={ // 스트리밍 상태를 ChatMessageComponent에 전달
                        message.role === 'assistant' &&
                        index === messages.length - 1 &&
                        isLoading &&
                        isReceivingResponse
                      }
                    />
                  ))}
                  {isLoading && !isReceivingResponse && (
                    <div className="flex items-start gap-3">
                      <Avatar className="h-8 w-8">
                        <AvatarImage src="/placeholder.svg?height=32&width=32" />
                        <AvatarFallback>AI</AvatarFallback>
                      </Avatar>
                      <div className="flex items-center mt-2">
                        <div className="relative h-8 w-8">
                          <div className="absolute inset-0 rounded-full border-4 border-slate-200 dark:border-slate-700"></div>
                          <div className="absolute inset-0 rounded-full border-4 border-t-blue-500 animate-spin"></div>
                        </div>
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
