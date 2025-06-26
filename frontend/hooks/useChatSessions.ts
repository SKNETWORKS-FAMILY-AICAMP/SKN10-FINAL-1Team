"use client"

import { useState, useEffect } from "react"
import type { TSession } from "@/types/chat"
import { API_BASE, getCookie, isJsonResponse } from "@/utils/api"

export function useChatSessions() {
  const [sessions, setSessions] = useState<TSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)

  // 디버깅용: 세션 배열에 중복 ID가 있는지 확인
  useEffect(() => {
    const ids = sessions.map((s) => s.id)
    const uniqueIds = new Set(ids)
    if (ids.length !== uniqueIds.size) {
      console.error("Duplicate session IDs detected!", sessions)
    }
    if (ids.some((id) => id === null || id === undefined)) {
      console.error("Null or undefined session ID detected!", sessions)
    }
  }, [sessions])

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const response = await fetch(`${API_BASE}/conversations/sessions/`)
        if (!response.ok || !isJsonResponse(response)) {
          throw new Error("Non-JSON or error response")
        }
        const data: TSession[] = await response.json()
        setSessions(data)
        if (data.length > 0) {
          setActiveSessionId(data[0].id)
        }
      } catch (error) {
        console.error("Error fetching sessions:", error)
        // 개발 환경에서는 더미 데이터 사용
        const dummySessions: TSession[] = [
          {
            id: "session-1",
            title: "새로운 대화",
            started_at: new Date().toISOString(),
          },
          {
            id: "session-2",
            title: "이전 대화",
            started_at: new Date().toISOString(),
          },
        ]
        setSessions(dummySessions)
        setActiveSessionId(dummySessions[0].id)
      }
    }

    fetchSessions()
  }, [])

  const createNewSession = async () => {
    try {
      const response = await fetch(`${API_BASE}/conversations/sessions/create/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken") || "",
        },
        body: JSON.stringify({ title: "새로운 채팅" }),
      })

      if (response.ok) {
        const newSession: TSession = await response.json()
        setSessions((prevSessions) => [newSession, ...prevSessions])
        setActiveSessionId(newSession.id)
        return newSession
      }
    } catch (error) {
      console.error("Error creating new session:", error)
      // 개발 환경에서는 로컬 상태만 업데이트
      const newSession: TSession = {
        id: `session-${Date.now()}`,
        title: "새로운 채팅",
        started_at: new Date().toISOString(),
      }
      setSessions((prevSessions) => [newSession, ...prevSessions])
      setActiveSessionId(newSession.id)
      return newSession
    }
  }

  const updateSessionTitle = (sessionId: string, title: string) => {
    setSessions((prevSessions) =>
      prevSessions.map((session) =>
        session.id === sessionId ? { ...session, title } : session
      )
    );
  };

  const deleteSession = async (sessionId: string) => {
    try {
      const response = await fetch(`${API_BASE}/conversations/sessions/${sessionId}/delete/`, {
        method: "DELETE",
        headers: {
          "X-CSRFToken": getCookie("csrftoken") || "",
        },
      })

      if (response.ok) {
        setSessions((prevSessions) => {
          const remainingSessions = prevSessions.filter((s) => s.id !== sessionId)
          if (activeSessionId === sessionId) {
            setActiveSessionId(remainingSessions.length > 0 ? remainingSessions[0].id : null)
          }
          return remainingSessions
        })
      }
    } catch (error) {
      console.error("Error deleting session:", error)
      // 개발 환경에서는 로컬 상태만 업데이트
      setSessions((prevSessions) => {
        const remainingSessions = prevSessions.filter((s) => s.id !== sessionId)
        if (activeSessionId === sessionId) {
          setActiveSessionId(remainingSessions.length > 0 ? remainingSessions[0].id : null)
        }
        return remainingSessions
      })
    }
  }

  return {
    sessions,
    activeSessionId,
    setActiveSessionId,
    createNewSession,
    deleteSession,
    updateSessionTitle,
  }
}
