// Mock data for preview/development environment
import type { User } from "./auth-service"
import type { ChatSession, ChatMessage, AgentType } from "./chat-service"
import { mockAgentResponse } from "../mock-agent"

// Mock user
export const mockUser: User = {
  id: "user123",
  org: {
    id: "org123",
    name: "Demo Organization",
  },
  email: "demo@example.com",
  name: "Demo User",
  role: "engineer",
  created_at: new Date().toISOString(),
  last_login: new Date().toISOString(),
  is_active: true,
  is_staff: false,
  avatar: "/placeholder.svg?height=40&width=40",
}

// Mock chat sessions
export const mockChatSessions: ChatSession[] = [
  {
    id: "session1",
    user: "user123",
    agent_type: "code",
    started_at: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(), // Yesterday
    ended_at: null,
    title: "Code Analysis Session",
  },
  {
    id: "session2",
    user: "user123",
    agent_type: "rag",
    started_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2).toISOString(), // 2 days ago
    ended_at: null,
    title: "Document Research",
  },
  {
    id: "session3",
    user: "user123",
    agent_type: "analytics",
    started_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(), // 1 week ago
    ended_at: null,
    title: "Business Metrics Review",
  },
]

// Mock messages for each session
export const mockMessages: Record<string, ChatMessage[]> = {
  session1: [
    {
      id: "msg1",
      session: "session1",
      role: "system",
      content: "Welcome to the Code Analysis session. How can I help you with your code today?",
      created_at: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
    },
  ],
  session2: [
    {
      id: "msg2",
      session: "session2",
      role: "system",
      content: "Welcome to the Document Research session. I can help you find information in your documents.",
      created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2).toISOString(),
    },
  ],
  session3: [
    {
      id: "msg3",
      session: "session3",
      role: "system",
      content: "Welcome to the Business Metrics Review session. I can help you analyze your business data.",
      created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(),
    },
  ],
}

// Helper function to check if we're in a preview/development environment
export function isPreviewEnvironment(): boolean {
  // Always return true for now since we don't have a backend
  return true

  // The original implementation below will be used when we have a real backend
  /*
  // Check if window is defined (we're in a browser)
  if (typeof window !== "undefined") {
    // Check for localhost or vercel preview domains
    const hostname = window.location.hostname
    return (
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname.includes("vercel.app") ||
      hostname.includes(".vercel.app")
    )
  }
  return false;
  */
}

// Mock function to generate a response based on user input
export function generateMockResponse(sessionId: string, content: string, agentType: AgentType): ChatMessage {
  // Use the existing mockAgentResponse function to generate a response
  const response = mockAgentResponse(content, agentType)

  return {
    id: `msg-${Date.now()}`,
    session: sessionId,
    role: "assistant",
    content: response.content,
    created_at: new Date().toISOString(),
    agentType: agentType,
    agentData: response.agentData,
  }
}
