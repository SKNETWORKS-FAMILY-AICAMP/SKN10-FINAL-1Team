"use client"

import type React from "react"

import { useState, useEffect, useRef, type FormEvent } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useChatSessions } from "@/hooks/useChatSessions"
import { useChatMessages } from "@/hooks/useChatMessages"
import type { TMessage, ChartContent, TToolCall, StreamMessage } from "@/types/chat"
import { API_BASE, getCookie } from "@/utils/api"
import { Sidebar } from "@/components/Sidebar"
import { ChatHeader } from "@/components/ChatHeader"
import { ChatMessage } from "@/components/ChatMessage"
import { LoadingIndicator } from "@/components/LoadingIndicator"
import { MessageInput } from "@/components/MessageInput"
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog"
import { ChartSidebar } from "@/components/ChartSidebar"

export default function ChatbotPage() {
  const { sessions, activeSessionId, setActiveSessionId, createNewSession, deleteSession, updateSessionTitle } = useChatSessions()
  const { messages, addMessage, updateMessage, clearMessages } = useChatMessages(activeSessionId)

  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [attachedFiles, setAttachedFiles] = useState<File[]>([])
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null)
  const [chartContent, setChartContent] = useState<ChartContent | null>(null)
  const [forceRefresh, setForceRefresh] = useState(false)
  const [isChartSidebarOpen, setChartSidebarOpen] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)

  // --- UI 효과 ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // 강제 리프레시 후 상태 초기화
  useEffect(() => {
    if (forceRefresh) {
      const timer = setTimeout(() => {
        setForceRefresh(false)
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [forceRefresh])

  // --- 이벤트 핸들러 ---
  const handleNewSession = async () => {
    await createNewSession()
    clearMessages()
  }

  const handleDeleteSession = (sessionId: string) => {
    setShowDeleteConfirm(sessionId)
  }

  const confirmDeleteSession = () => {
    if (showDeleteConfirm) {
      deleteSession(showDeleteConfirm)
      if (activeSessionId === showDeleteConfirm) {
        clearMessages()
      }
    }
    setShowDeleteConfirm(null)
  }

  const handleFileAttach = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (files) {
      const newFiles = Array.from(files).slice(0, 5)
      setAttachedFiles((prev) => [...prev, ...newFiles].slice(0, 5))
    }
    event.target.value = ""
  }

  const removeAttachedFile = (index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || !activeSessionId) return

    const userMessage: TMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: input,
      createdAt: new Date().toISOString(),
    }

    addMessage(userMessage)
    setInput("")
    setAttachedFiles([])
    setIsLoading(true)

    try {
      const headers: HeadersInit = {
        "X-CSRFToken": getCookie("csrftoken") || "",
      };
      let body: BodyInit;

      if (attachedFiles.length > 0) {
        const formData = new FormData();
        formData.append("message", userMessage.content);
        attachedFiles.forEach((file) => {
          formData.append("files", file, file.name);
        });
        body = formData;
      } else {
        headers["Content-Type"] = "application/json";
        body = JSON.stringify({ message: userMessage.content });
      }

      const response = await fetch(`${API_BASE}/conversations/sessions/${activeSessionId}/invoke/`, {
        method: "POST",
        headers,
        body,
      });

      if (!response.ok || response.body == null) {
        throw new Error("Streaming endpoint unavailable")
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      const assistantMessageId = `assistant-${Date.now()}`
      const currentAiMessage: TMessage = {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        createdAt: new Date().toISOString(),
        tool_calls: [],
      }

      addMessage(currentAiMessage)

      let accumulatedContent = ""
      let accumulatedTitle = ""
      let isTitleStarted = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split("\n\n")

        for (const line of lines) {
          if (line.startsWith("data:")) {
            try {
              const jsonData = line.substring(5).trim()
              if (!jsonData) continue
              
              // JSON 파싱 전에 기본적인 유효성 검사
              if (!jsonData.startsWith('{') && !jsonData.startsWith('[')) {
                console.warn("Invalid JSON format, skipping:", jsonData.substring(0, 100))
                continue
              }
              
              const json = JSON.parse(jsonData)
              if (json.event === "message_chunk") {
                accumulatedContent += json.data || ""
                updateMessage(assistantMessageId, {
                  content: accumulatedContent,
                })
              } else if (json.event === "tool_update") {
                const assistantMessages: StreamMessage[] = json.data.assistant.messages || []

                // 1. Consolidate all tool calls from 'ai' messages
                const allToolCalls = assistantMessages
                  .filter((msg): msg is StreamMessage & { tool_calls: TToolCall[] } => 
                    msg.type === "ai" && "tool_calls" in msg
                  )
                  .flatMap((msg) => msg.tool_calls)

                // 2. Create a map of tool outputs from 'tool' messages
                const toolOutputs = new Map<string, string>()
                assistantMessages
                  .filter((msg: StreamMessage) => msg.type === "tool")
                  .forEach((msg: StreamMessage) => {
                    if (msg.tool_call_id && msg.content) {
                    toolOutputs.set(msg.tool_call_id, msg.content)
                    }
                  })

                // 3. Combine calls and outputs into the final TToolCall array
                const finalToolCalls: TToolCall[] = allToolCalls.map((call: TToolCall) => ({
                  ...call,
                  output: toolOutputs.get(call.id) || null,
                }))

                // 4. Update the message state
                updateMessage(assistantMessageId, {
                  tool_calls: finalToolCalls,
                })

                // 5. Handle chart data from the final tool calls
                const chartToolCall = finalToolCalls.find(
                  (call: TToolCall) =>
                    (call.name === "ChartGenerator" || call.name === "analyst_chart_tool") && call.output,
                )

                if (chartToolCall && typeof chartToolCall.output === "string") {
                  try {
                    // The output is already a string, which should be JSON
                    const chartData = JSON.parse(chartToolCall.output)
                    setChartContent(chartData)
                  } catch (error) {
                    console.error("Failed to parse chart data:", error)
                  }
                }

                // 6. 디버깅 로그 추가
                console.log(`🔧 Tool update received: ${finalToolCalls.length} tools`)
                finalToolCalls.forEach((call, index) => {
                  console.log(`  ${index + 1}. ${call.name}: ${call.output ? '완료' : '진행 중'}`)
                })
              } else if (json.event === "title_start") {
                isTitleStarted = true
                if (activeSessionId) {
                  updateSessionTitle(activeSessionId, "") // Clear title on start
                }
              } else if (json.event === "title_chunk") {
                if (isTitleStarted) {
                  accumulatedTitle += json.data || ""
                  if (activeSessionId) {
                    updateSessionTitle(activeSessionId, accumulatedTitle)
                  }
                }
              } else if (json.event === "stream_end") {
                setIsLoading(false)
                // 스트리밍 완료 시 강제 리프레시 트리거
                setForceRefresh(true)
                console.log("🔄 스트리밍 완료 - ToolCallDisplay 강제 리프레시 트리거")
                
                // 강제 리프레시 후 더 긴 지연을 두고 상태 재확인
                setTimeout(() => {
                  console.log("🔍 강제 리프레시 후 도구 상태 재확인")
                  setForceRefresh(false)
                }, 500) // 200ms에서 500ms로 증가
              }
            } catch (error) {
              if (process.env.NODE_ENV === "development") {
                console.warn("Stream parse error", error)
                console.warn("Problematic line:", line.substring(0, 200))
                // JSON 파싱 오류가 발생해도 스트리밍을 계속 진행
                continue
              }
            }
          }
        }
      }
      setIsLoading(false); // Ensure loading is always stopped
    } catch (error) {
      console.error("Error sending message:", error)
      setIsLoading(false)
      // 개발 환경에서는 더미 응답
      setTimeout(() => {
        const botResponse: TMessage = {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: "죄송합니다. 현재 Django 백엔드와 연결되지 않았습니다. API 서버를 확인해주세요.",
          createdAt: new Date().toISOString(),
        }
        addMessage(botResponse)
        setIsLoading(false)
      }, 1000)
    }
  }

  return (
    <div className="flex h-screen bg-gray-50 relative">
      {/* 사이드바 오버레이 (모바일용) */}
      {isSidebarOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden" onClick={() => setIsSidebarOpen(false)} />
      )}

      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSessionSelect={setActiveSessionId}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
      />

      {/* 메인 채팅 영역 */}
      <div
        className={`flex-1 flex flex-col min-w-0 transition-all duration-300 ${isSidebarOpen ? "lg:ml-80" : "ml-0"}`}
      >
        <ChatHeader
          isSidebarOpen={isSidebarOpen}
          onToggleSidebar={() => setIsSidebarOpen(true)}
          sessions={sessions}
          activeSessionId={activeSessionId}
        />

        {/* 메시지 영역 */}
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4 max-w-4xl mx-auto">
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                onOpenChart={(chartData?: ChartContent) => {
                  if (chartData) {
                  setChartContent(chartData);
                  setChartSidebarOpen(true);
                  }
                }}
                forceRefresh={forceRefresh}
              />
            ))}

            {isLoading && <LoadingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        <MessageInput
          input={input}
          onInputChange={setInput}
          onSubmit={handleSubmit}
          isLoading={isLoading}
          isDisabled={!activeSessionId}
          attachedFiles={attachedFiles}
          onFileAttach={handleFileAttach}
          onRemoveFile={removeAttachedFile}
        />
      </div>

      <ChartSidebar
        isOpen={isChartSidebarOpen}
        onClose={() => setChartSidebarOpen(false)}
        chartContent={chartContent}
      />

      <DeleteConfirmDialog
        isOpen={!!showDeleteConfirm}
        onConfirm={confirmDeleteSession}
        onCancel={() => setShowDeleteConfirm(null)}
      />
    </div>
  )
}