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

// 백엔드 가용성 확인을 위한 변수들
let _isBackendChecked = false;
let _isBackendAvailable = false;

// 백엔드 연결 확인 함수
async function checkBackendConnection(): Promise<boolean> {
  if (_isBackendChecked) return _isBackendAvailable;
  
  try {
    const response = await fetch("/api/health-check/", {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      redirect: 'follow', // 리다이렉트 자동 처리
    });
    
    if (response.ok) {
      console.log("Backend connection successful");
      localStorage.setItem("backendAvailable", "true");
      localStorage.removeItem("demoMode");
      _isBackendAvailable = true;
    } else {
      console.log("Backend returned error status");
      localStorage.setItem("demoMode", "true");
      localStorage.removeItem("backendAvailable");
      _isBackendAvailable = false;
    }
  } catch (error) {
    console.log("Backend connection failed:", error);
    localStorage.setItem("demoMode", "true");
    localStorage.removeItem("backendAvailable");
    _isBackendAvailable = false;
  }
  
  _isBackendChecked = true;
  return _isBackendAvailable;
}

// 초기 연결 확인 시작 (페이지 로드 시 한 번만 실행)
if (typeof window !== "undefined") {
  checkBackendConnection();
}

// Helper function to check if we're in a preview/environment
export function isPreviewEnvironment(): boolean {
  // Check if window is defined (we're in a browser)
  if (typeof window !== "undefined") {
    console.log("✅ isPreviewEnvironment 호출됨", {
      backendChecked: _isBackendChecked,
      backendAvailable: _isBackendAvailable,
      localStorageDemoMode: localStorage.getItem("demoMode"),
      localStorageBackendAvailable: localStorage.getItem("backendAvailable"),
      pathname: window.location.pathname
    });
    
    // 이미 데모 모드로 설정되어 있는지 확인
    const demoMode = localStorage.getItem("demoMode") === "true";
    if (demoMode) {
      console.log("✅ 데모 모드 활성화 - mock 데이터 사용");
      return true;
    }
    
    // 백엔드 가용성이 확인되어 있으면 그 결과 사용
    const backendAvailable = localStorage.getItem("backendAvailable") === "true";
    if (backendAvailable) {
      console.log("✅ 백엔드 연결 확인됨 - 실제 API 사용");
      return false;
    }
    
    // 백엔드 연결 강제 사용 설정 (디버깅용)
    // 로컬 스토리지에 forceRealApi=true를 설정하면 항상 실제 API 호출
    const forceRealApi = localStorage.getItem("forceRealApi") === "true";
    if (forceRealApi) {
      console.log("✅ API 강제 사용 모드 - 실제 API 사용");
      localStorage.setItem("backendAvailable", "true");
      return false;
    }
    
    // 아직 확인되지 않았고 데모 페이지가 아니면 항상 mock 데이터 반환
    // 실제 백엔드 연결은 비동기적으로 이루어지므로 첫 요청은 항상 mock 데이터를 사용
    if (!window.location.pathname.includes("/demo") && !_isBackendChecked) {
      console.log("✅ 백엔드 연결 확인 중 - 임시 mock 데이터 사용");
      return true;
    }
    
    // 백엔드 확인 완료된 상태면 실제 가용성 결과 반환
    const result = !_isBackendAvailable;
    console.log(`✅ 백엔드 확인 완료 - ${result ? "mock" : "실제"} 데이터 사용`);
    return result;
  }
  return false;
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
