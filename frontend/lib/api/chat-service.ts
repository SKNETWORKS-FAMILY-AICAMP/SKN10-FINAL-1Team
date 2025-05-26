// Chat service for interacting with Django backend
// Connects to the ChatSession and ChatMessage models in the Conversations section

import { toast } from "@/hooks/use-toast"
import { isPreviewEnvironment, mockChatSessions, mockMessages, generateMockResponse } from "./mock-data"

// Types that match Django models
export type AgentType = "code" | "rag" | "analytics" // Matches AgentType enum in schema

export interface ChatSession {
  id: string
  user: string // User ID
  agent_type: AgentType
  started_at: string
  ended_at: string | null
  title?: string // Not in schema but useful for UI
}

export interface ChatMessage {
  id: string
  session: string // Session ID
  role: "user" | "assistant" | "system"
  content: string
  created_at: string
  // Additional fields for UI rendering
  agentType?: AgentType
  agentData?: any // For structured data responses
}

// Base URL for API calls
const API_BASE_URL = "/api"

/**
 * Get authentication headers for API calls
 */
function getAuthHeaders() {
  const token = localStorage.getItem("token")
  return {
    "Content-Type": "application/json",
    Authorization: `Token ${token}`,
  }
}

/**
 * Get all chat sessions for the current user
 * Maps to Django ChatSession model filtered by user
 */
export async function getChatSessions(): Promise<ChatSession[]> {
  try {
    // Always use mock data until backend is connected
    if (isPreviewEnvironment()) {
      return Promise.resolve([...mockChatSessions])
    }

    const response = await fetch(`${API_BASE_URL}/chat/sessions/`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error("Failed to fetch chat sessions")
    }

    return await response.json()
  } catch (error) {
    console.error("Error fetching chat sessions:", error)

    // Fallback to mock data if API call fails
    console.log("Falling back to mock data")
    return [...mockChatSessions]
  }
}

/**
 * Create a new chat session
 * Creates a new record in the ChatSession table
 */
export async function createChatSession(agentType: AgentType): Promise<ChatSession | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    const newSession: ChatSession = {
      id: `session-${Date.now()}`,
      user: "user123",
      agent_type: agentType,
      started_at: new Date().toISOString(),
      ended_at: null,
      title: `New ${agentType.charAt(0).toUpperCase() + agentType.slice(1)} Session`,
    }

    // Add to mock sessions
    mockChatSessions.unshift(newSession)

    // Initialize empty message array for this session
    mockMessages[newSession.id] = [
      {
        id: `welcome-${Date.now()}`,
        session: newSession.id,
        role: "system",
        content: `Welcome to your new ${agentType} session. How can I help you today?`,
        created_at: new Date().toISOString(),
      },
    ]

    return Promise.resolve(newSession)
  }

  try {
    const response = await fetch(`${API_BASE_URL}/chat/sessions/`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ agent_type: agentType }),
    })

    if (!response.ok) {
      throw new Error("Failed to create chat session")
    }

    return await response.json()
  } catch (error) {
    console.error("Error creating chat session:", error)
    toast({
      title: "Error",
      description: "Failed to create new chat session",
      variant: "destructive",
    })
    return null
  }
}

/**
 * Get messages for a specific chat session
 * Maps to Django ChatMessage model filtered by session
 */
export async function getChatMessages(sessionId: string): Promise<ChatMessage[]> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    return Promise.resolve(mockMessages[sessionId] || [])
  }

  try {
    const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/messages/`, {
      headers: getAuthHeaders(),
    })

    if (!response.ok) {
      throw new Error("Failed to fetch chat messages")
    }

    return await response.json()
  } catch (error) {
    console.error("Error fetching chat messages:", error)
    toast({
      title: "Error",
      description: "Failed to load chat messages",
      variant: "destructive",
    })
    return []
  }
}

/**
 * Send a message to the chat session
 * Creates a new ChatMessage and triggers LLM processing on the backend
 * The backend will create LlmCall records for tracking
 */
export async function sendChatMessage(sessionId: string, content: string): Promise<ChatMessage | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    // Find the session to get its agent type
    const session = mockChatSessions.find((s) => s.id === sessionId)
    if (!session) return null

    // Create user message
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      session: sessionId,
      role: "user",
      content: content,
      created_at: new Date().toISOString(),
    }

    // Add user message to mock messages
    if (!mockMessages[sessionId]) {
      mockMessages[sessionId] = []
    }
    mockMessages[sessionId].push(userMessage)

    // Generate AI response
    const aiResponse = generateMockResponse(sessionId, content, session.agent_type)

    // Add AI response to mock messages
    mockMessages[sessionId].push(aiResponse)

    return Promise.resolve(aiResponse)
  }

  try {
    const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/messages/`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ content }),
    })

    if (!response.ok) {
      throw new Error("Failed to send message")
    }

    return await response.json()
  } catch (error) {
    console.error("Error sending chat message:", error)
    toast({
      title: "Error",
      description: "Failed to send message",
      variant: "destructive",
    })
    return null
  }
}

/**
 * End a chat session
 * Updates the ended_at field in the ChatSession model
 */
export async function endChatSession(sessionId: string): Promise<boolean> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    const sessionIndex = mockChatSessions.findIndex((s) => s.id === sessionId)
    if (sessionIndex !== -1) {
      mockChatSessions[sessionIndex].ended_at = new Date().toISOString()
      return Promise.resolve(true)
    }
    return Promise.resolve(false)
  }

  try {
    const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/`, {
      method: "PATCH",
      headers: getAuthHeaders(),
      body: JSON.stringify({ ended_at: new Date().toISOString() }),
    })

    if (!response.ok) {
      throw new Error("Failed to end chat session")
    }

    return true
  } catch (error) {
    console.error("Error ending chat session:", error)
    toast({
      title: "Error",
      description: "Failed to end chat session",
      variant: "destructive",
    })
    return false
  }
}
