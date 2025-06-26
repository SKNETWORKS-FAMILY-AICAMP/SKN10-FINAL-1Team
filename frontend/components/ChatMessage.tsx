"use client"

import { Bot, BarChart3 } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { TMessage, ChartContent } from "@/types/chat"
import { ToolCallDisplay } from "./ToolCallDisplay"

interface ChatMessageProps {
  message: TMessage
  chartContent: ChartContent | null
  onOpenChart: () => void
  forceRefresh?: boolean
}

export function ChatMessage({ message, chartContent, onOpenChart, forceRefresh = false }: ChatMessageProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-xl">
          <div className="bg-gradient-to-r from-blue-600 to-blue-500 text-white p-4 rounded-2xl rounded-tr-none shadow-lg">
            <p className="text-sm">{message.content}</p>
          </div>
          <p className="text-xs text-gray-500 text-right mt-1">{new Date(message.createdAt).toLocaleTimeString()}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-start gap-3">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-gray-100 to-gray-200 text-gray-600 flex items-center justify-center shadow-sm">
        <Bot className="h-4 w-4" />
      </div>
      <div className="flex-1 max-w-3xl">
        <ToolCallDisplay toolCalls={message.tool_calls || []} forceRefresh={forceRefresh} />

        <div className="bg-white text-gray-800 p-4 rounded-2xl rounded-tl-none shadow-lg border border-gray-100">
          <div className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</div>
        </div>
        <p className="text-xs text-gray-500 mt-1">{new Date(message.createdAt).toLocaleTimeString()}</p>

        {chartContent && (
          <div className="mt-3">
            <Button
              onClick={onOpenChart}
              size="sm"
              className="bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white shadow-md hover:shadow-lg transition-all duration-200"
            >
              <BarChart3 className="h-4 w-4 mr-2" />
              차트 보기
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
