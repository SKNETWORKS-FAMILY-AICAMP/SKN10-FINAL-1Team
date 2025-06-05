/**
 * PostgreSQL Chat Service
 * Provides direct database access to PostgreSQL for chat sessions and messages
 * Bypasses the Django backend for chat data handling
 */

import { v4 as uuidv4 } from 'uuid'
import { toast } from "@/hooks/use-toast"
import { AgentType, ChatSession, ChatMessage } from "./chat-service"
import { isPreviewEnvironment, mockChatSessions, mockMessages } from "./mock-data"

// Connection string for PostgreSQL - should be set in environment variables
const PG_CONNECTION_STRING = process.env.NEXT_PUBLIC_PG_CONNECTION_STRING || 
  "postgresql://username:password@localhost:5432/sknetworks"

// Helper function to execute database queries
async function executeQuery(sql: string, params: any[] = []): Promise<any> {
  // When in preview/development environment, return mock data instead
  if (isPreviewEnvironment()) {
    console.log("Using mock data in preview environment")
    return null
  }

  try {
    // Get access token from localStorage for authentication
    const accessToken = typeof window !== 'undefined' ? localStorage.getItem('accessToken') : null;
    
    // Prepare headers with authentication if token exists
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    
    if (accessToken) {
      headers['Authorization'] = `Bearer ${accessToken}`;
    }
    
    // Execute query through our secure API route
    const response = await fetch('/api/db', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        sql,
        params
      }),
    })

    if (!response.ok) {
      throw new Error(`Database query failed: ${response.statusText}`)
    }

    return await response.json()
  } catch (error) {
    console.error("Database query error:", error)
    toast({
      title: "Database Error",
      description: "Failed to execute database query",
      variant: "destructive",
    })
    throw error
  }
}

/**
 * Get all chat sessions for a user
 */
export async function getUserChatSessions(userId: string): Promise<ChatSession[]> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    return Promise.resolve([...mockChatSessions])
  }

  try {
    const sql = `
      SELECT id, user_id, agent_type, started_at, ended_at, title
      FROM chat_sessions
      WHERE user_id = $1
      ORDER BY started_at DESC
    `
    const result = await executeQuery(sql, [userId])
    
    // Map the database results to match our ChatSession interface
    return result.rows.map((row: any) => ({
      id: row.id,
      user: row.user_id,
      agent_type: row.agent_type,
      started_at: row.started_at,
      ended_at: row.ended_at,
      title: row.title
    }))
  } catch (error) {
    console.error("Error fetching chat sessions:", error)
    // Fallback to mock data if API call fails
    return [...mockChatSessions]
  }
}

/**
 * Create a new chat session in the database
 */
export async function createChatSessionInDB(userId: string, agentType: AgentType): Promise<ChatSession | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    const newSession: ChatSession = {
      id: `session-${Date.now()}`,
      user: userId,
      agent_type: agentType,
      started_at: new Date().toISOString(),
      ended_at: null,
      title: `New ${agentType.charAt(0).toUpperCase() + agentType.slice(1)} Session`,
    }
    mockChatSessions.unshift(newSession)
    return Promise.resolve(newSession)
  }

  try {
    const sessionId = uuidv4()
    const now = new Date().toISOString()
    const title = `New ${agentType.charAt(0).toUpperCase() + agentType.slice(1)} Session`
    
    const sql = `
      INSERT INTO chat_sessions (id, user_id, agent_type, started_at, title)
      VALUES ($1, $2, $3, $4, $5)
      RETURNING id, user_id, agent_type, started_at, ended_at, title
    `
    
    const result = await executeQuery(sql, [sessionId, userId, agentType, now, title])
    
    if (result && result.rows && result.rows.length > 0) {
      const session = result.rows[0]
      return {
        id: session.id,
        user: session.user_id,
        agent_type: session.agent_type,
        started_at: session.started_at,
        ended_at: session.ended_at,
        title: session.title
      }
    }
    
    return null
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
 */
export async function getChatMessagesFromDB(sessionId: string): Promise<ChatMessage[]> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    return Promise.resolve(mockMessages[sessionId] || [])
  }

  try {
    const sql = `
      SELECT id, session_id, role, content, created_at
      FROM chat_messages
      WHERE session_id = $1
      ORDER BY created_at ASC
    `
    
    const result = await executeQuery(sql, [sessionId])
    
    // Map the database results to match our ChatMessage interface
    return result.rows.map((row: any) => ({
      id: row.id,
      session: row.session_id,
      role: row.role,
      content: row.content,
      created_at: row.created_at
    }))
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
 * Save a user message to the database
 */
export async function saveChatMessageToDB(
  sessionId: string, 
  role: "user" | "assistant" | "system", 
  content: string
): Promise<ChatMessage | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    const newMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      session: sessionId,
      role,
      content,
      created_at: new Date().toISOString(),
    }
    
    if (!mockMessages[sessionId]) {
      mockMessages[sessionId] = []
    }
    mockMessages[sessionId].push(newMessage)
    
    return Promise.resolve(newMessage)
  }

  try {
    const messageId = uuidv4()
    const now = new Date().toISOString()
    
    const sql = `
      INSERT INTO chat_messages (id, session_id, role, content, created_at)
      VALUES ($1, $2, $3, $4, $5)
      RETURNING id, session_id, role, content, created_at
    `
    
    const result = await executeQuery(sql, [messageId, sessionId, role, content, now])
    
    if (result && result.rows && result.rows.length > 0) {
      const message = result.rows[0]
      return {
        id: message.id,
        session: message.session_id,
        role: message.role,
        content: message.content,
        created_at: message.created_at
      }
    }
    
    return null
  } catch (error) {
    console.error("Error saving chat message:", error)
    toast({
      title: "Error",
      description: "Failed to save message",
      variant: "destructive",
    })
    return null
  }
}

/**
 * Update the agent type for a chat session
 */
export async function updateSessionAgentTypeInDB(sessionId: string, agentType: AgentType): Promise<ChatSession | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    const sessionIndex = mockChatSessions.findIndex((s) => s.id === sessionId)
    if (sessionIndex !== -1) {
      mockChatSessions[sessionIndex].agent_type = agentType
      return Promise.resolve({ ...mockChatSessions[sessionIndex] })
    }
    return Promise.resolve(null)
  }

  try {
    const sql = `
      UPDATE chat_sessions
      SET agent_type = $1
      WHERE id = $2
      RETURNING id, user_id, agent_type, started_at, ended_at, title
    `
    
    const result = await executeQuery(sql, [agentType, sessionId])
    
    if (result && result.rows && result.rows.length > 0) {
      const session = result.rows[0]
      return {
        id: session.id,
        user: session.user_id,
        agent_type: session.agent_type,
        started_at: session.started_at,
        ended_at: session.ended_at,
        title: session.title
      }
    }
    
    return null
  } catch (error) {
    console.error("Error updating session agent type:", error)
    toast({
      title: "Error",
      description: "Failed to update agent type",
      variant: "destructive",
    })
    return null
  }
}

/**
 * End a chat session
 */
export async function endChatSessionInDB(sessionId: string): Promise<boolean> {
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
    const now = new Date().toISOString()
    
    const sql = `
      UPDATE chat_sessions
      SET ended_at = $1
      WHERE id = $2
    `
    
    await executeQuery(sql, [now, sessionId])
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

/**
 * Delete a chat session
 */
export async function deleteChatSessionFromDB(sessionId: string): Promise<boolean> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    const sessionIndex = mockChatSessions.findIndex((s) => s.id === sessionId)
    if (sessionIndex !== -1) {
      mockChatSessions.splice(sessionIndex, 1)
      return Promise.resolve(true)
    }
    return Promise.resolve(false)
  }

  try {
    // First delete related llm_calls records
    let sql = `
      DELETE FROM llm_calls
      WHERE session_id = $1
    `
    await executeQuery(sql, [sessionId])
    
    // Then delete all messages in the session
    sql = `
      DELETE FROM chat_messages
      WHERE session_id = $1
    `
    await executeQuery(sql, [sessionId])
    
    // Finally delete the session itself
    sql = `
      DELETE FROM chat_sessions
      WHERE id = $1
    `
    await executeQuery(sql, [sessionId])
    
    toast({
      title: "Session Deleted",
      description: "The chat session has been deleted successfully",
    })
    return true
  } catch (error) {
    console.error("Error deleting chat session:", error)
    toast({
      title: "Error",
      description: "Failed to delete chat session",
      variant: "destructive",
    })
    return false
  }
}

/**
 * Update a chat session title
 */
export async function updateChatSessionTitleInDB(sessionId: string, title: string): Promise<ChatSession | null> {
  // Use mock data in preview environment
  if (isPreviewEnvironment()) {
    const sessionIndex = mockChatSessions.findIndex((s) => s.id === sessionId)
    if (sessionIndex !== -1) {
      mockChatSessions[sessionIndex].title = title
      return Promise.resolve({ ...mockChatSessions[sessionIndex] })
    }
    return Promise.resolve(null)
  }

  try {
    const sql = `
      UPDATE chat_sessions
      SET title = $1
      WHERE id = $2
      RETURNING id, user_id, agent_type, started_at, ended_at, title
    `
    
    const result = await executeQuery(sql, [title, sessionId])
    
    if (result && result.rows && result.rows.length > 0) {
      const session = result.rows[0]
      return {
        id: session.id,
        user: session.user_id,
        agent_type: session.agent_type,
        started_at: session.started_at,
        ended_at: session.ended_at,
        title: session.title
      }
    }
    
    return null
  } catch (error) {
    console.error("Error updating session title:", error)
    toast({
      title: "Error",
      description: "Failed to update session title",
      variant: "destructive",
    })
    return null
  }
}
