/**
 * FastAPI service for interacting with the FastAPI server
 * Provides functionality for both REST and WebSocket interactions
 */

import { toast } from "@/hooks/use-toast"
import { v4 as uuidv4 } from 'uuid'
import { cleanStreamingToken, cleanFullMessageContent } from './token-cleaner'

// Set the FastAPI server URL
const FASTAPI_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://localhost:8001"
const FASTAPI_WS_URL = FASTAPI_URL.replace('http://', 'ws://').replace('https://', 'wss://')

export interface ChatRequest {
  message: string;
  thread_id?: string;
}

export interface ChatResponse {
  thread_id: string;
  message: {
    role: string;
    content: string;
  };
}

export interface StreamEvent {
  event_type: 'token' | 'agent_change' | 'tool_start' | 'tool_end' | 'error' | 'done';
  data: any;
}

// For tracking active WebSocket connections
const activeConnections: Record<string, WebSocket> = {}

/**
 * Send a message to the FastAPI server and receive a full response
 */
export async function sendChatMessageToFastAPI(message: string, threadId?: string): Promise<ChatResponse | null> {
  try {
    const response = await fetch(`${FASTAPI_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ 
        message, 
        thread_id: threadId 
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to send message: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error sending message to FastAPI:", error);
    toast({
      title: "Error",
      description: "Failed to send message to FastAPI server",
      variant: "destructive",
    });
    return null;
  }
}

/**
 * Create a WebSocket connection to the FastAPI server for streaming chat responses
 * Returns an object with methods to send messages, close the connection, and the connection itself
 */
export function createChatWebSocket(threadId: string = uuidv4(), 
  onToken: (token: string) => void,
  onAgentChange: (agent: string) => void,
  onToolStart: (tool: any) => void,
  onToolEnd: (result: any) => void,
  onError: (error: string) => void,
  onDone: () => void) {
  
  console.log(`Creating WebSocket connection for thread ${threadId}`);
  console.log(`FastAPI WebSocket URL: ${FASTAPI_WS_URL}/api/chat/ws/${threadId}`);
  
  try {
    // Close existing connection for this thread if any
    if (activeConnections[threadId]) {
      try {
        activeConnections[threadId].close();
      } catch (e) {
        console.warn("Error closing existing WebSocket:", e);
      }
    }
    
    // Create a new WebSocket connection
    const ws = new WebSocket(`${FASTAPI_WS_URL}/api/chat/ws/${threadId}`);
    
    // Store the connection
    activeConnections[threadId] = ws;
    
    // Set up event handlers
    ws.onopen = () => {
      console.log(`WebSocket connection established for thread ${threadId}`);
    };
    
    ws.onerror = (error) => {
      console.error(`WebSocket error for thread ${threadId}:`, error);
      onError(`WebSocket connection error. Please check if FastAPI server is running on ${FASTAPI_WS_URL}`);
    };
    
    ws.onclose = (event) => {
      console.log(`WebSocket closed for thread ${threadId}: ${event.code} ${event.reason}`);
      delete activeConnections[threadId];
      // 연결이 기대하지 않고 끊길 때는 오류 메시지를 표시하는 것이 좋습니다
      if (event.code !== 1000) {
        onError(`WebSocket connection closed unexpectedly. Code: ${event.code}`);
      }
    };
    
    ws.onmessage = (event) => {
      try {
        const streamEvent: StreamEvent = JSON.parse(event.data);
        
        switch (streamEvent.event_type) {
          case 'token':
            // Clean the token to remove routing directives before passing to the callback
            const token = streamEvent.data.token;
            const cleanedToken = cleanStreamingToken(token);
            
            // Only send non-empty tokens
            if (cleanedToken) {
              onToken(cleanedToken);
            }
            break;
            
          case 'agent_change':
            onAgentChange(streamEvent.data.agent);
            break;
            
          case 'tool_start':
            onToolStart(streamEvent.data);
            break;
            
          case 'tool_end':
            onToolEnd(streamEvent.data);
            break;
            
          case 'error':
            onError(streamEvent.data.error);
            break;
            
          case 'done':
            // If there's complete content available, clean it before finalizing
            if (streamEvent.data && streamEvent.data.content) {
              streamEvent.data.content = cleanFullMessageContent(streamEvent.data.content);
            }
            onDone();
            break;
            
          default:
            console.warn("Unknown event type:", streamEvent.event_type);
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error, event.data);
      }
    };
    
    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      toast({
        title: "Connection Error",
        description: "Failed to establish WebSocket connection to FastAPI server",
        variant: "destructive",
      });
    };
    
    ws.onclose = () => {
      console.log(`WebSocket connection closed for thread ${threadId}`);
      delete activeConnections[threadId];
    };
    
    // Return methods for interacting with the WebSocket
    return {
      sendMessage: (message: string, csvFileContent?: string) => {
        // 연결 상태 확인 및 재연결 시도
        if (ws.readyState === WebSocket.OPEN) {
          console.log(`Sending message over WebSocket: ${message.substring(0, 30)}...`);
          const payload: { message: string; csv_file_content?: string } = { message };
          if (csvFileContent) {
            payload.csv_file_content = csvFileContent;
          }
          ws.send(JSON.stringify(payload));
          return true;
        } else if (ws.readyState === WebSocket.CONNECTING) {
          console.log("WebSocket is still connecting, waiting...");
          // 연결 대기 후 재시도
          setTimeout(() => {
            if (ws.readyState === WebSocket.OPEN) {
              console.log(`Retry sending message over WebSocket after connecting`);
              const payload: { message: string; csv_file_content?: string } = { message };
              if (csvFileContent) {
                payload.csv_file_content = csvFileContent;
              }
              ws.send(JSON.stringify(payload));
            } else {
              console.error("WebSocket failed to connect in time");
              onError("Failed to establish connection with the AI assistant. Please try again.");
            }
          }, 1000); // 1초 대기
          return true;
        } else {
          console.error(`WebSocket is not open (readyState: ${ws.readyState})`);
          onError("Connection to AI assistant lost. Please refresh and try again.");
          return false;
        }
      },
      
      close: () => {
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
          ws.close();
        }
        delete activeConnections[threadId];
      },
      
      ws,
      threadId
    };
  } catch (error) {
    console.error("Error creating WebSocket:", error);
    toast({
      title: "Connection Error",
      description: "Failed to establish WebSocket connection",
      variant: "destructive",
    });
    return null;
  }
}

/**
 * Close all active WebSocket connections
 * Useful when navigating away or cleaning up
 */
export function closeAllWebSockets() {
  Object.values(activeConnections).forEach(ws => {
    try {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    } catch (e) {
      console.warn("Error closing WebSocket:", e);
    }
  });
  
  // Clear the connections object
  Object.keys(activeConnections).forEach(key => {
    delete activeConnections[key];
  });
}
