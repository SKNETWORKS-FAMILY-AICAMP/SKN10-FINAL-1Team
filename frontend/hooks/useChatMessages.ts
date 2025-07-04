"use client"

import { useState, useEffect } from "react"
import type { TMessage } from "@/types/chat"
import { API_BASE, isJsonResponse } from "@/utils/api"

export function useChatMessages(activeSessionId: string | null) {
  const [messages, setMessages] = useState<TMessage[]>([])

  useEffect(() => {
    if (activeSessionId) {
      const fetchMessages = async () => {
        try {
          const response = await fetch(`${API_BASE}/conversations/sessions/${activeSessionId}/messages/`)
          if (!response.ok || !isJsonResponse(response)) {
            throw new Error("Non-JSON or error response")
          }
          const data: TMessage[] = await response.json()
          setMessages(data)
        } catch (error) {
          console.error("Error fetching messages:", error)
          // 개발 환경에서는 더미 메시지 사용
          setMessages([
            {
              id: "1",
              role: "assistant",
              content: "안녕하세요! AI 챗봇입니다. 무엇을 도와드릴까요?",
              createdAt: new Date().toISOString(),
            },
          ])
        }
      }

      fetchMessages()
    }
  }, [activeSessionId])

  const addMessage = (message: TMessage) => {
    setMessages((prev) => [...prev, message])
  }

  const updateMessage = (messageId: string, updates: Partial<TMessage>) => {
    setMessages((prev) => prev.map((msg) => (msg.id === messageId ? { ...msg, ...updates } : msg)))
  }

  const clearMessages = () => {
    setMessages([])
  }

  return {
    messages,
    setMessages,
    addMessage,
    updateMessage,
    clearMessages,
  }
}
