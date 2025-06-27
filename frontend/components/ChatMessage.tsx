"use client"

import { Bot, BarChart3 } from "lucide-react"
import { Button } from "@/components/ui/button"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import type { TMessage, ChartContent } from "@/types/chat"
import { ToolCallDisplay } from "./ToolCallDisplay"
import { useMemo } from "react"

interface ChatMessageProps {
  message: TMessage
  chartContent: ChartContent | null
  onOpenChart: (data?: any) => void
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

  // 차트 데이터 감지 (output이 string이든 object든 모두 지원)
  const chartToolCall = useMemo(() => {
    if (!message.tool_calls) return null;
    return message.tool_calls.find((call) => {
      if (call.name !== "ChartGenerator" && call.name !== "analyst_chart_tool") return false;
      if (!call.output) return false;
      let parsed = null;
      if (typeof call.output === "string") {
        try {
          parsed = JSON.parse(call.output);
        } catch {
          return false;
        }
      } else if (typeof call.output === "object" && call.output !== null) {
        parsed = call.output;
      }
      return parsed && parsed.canvas_html && parsed.script_js;
    });
  }, [message.tool_calls]);

  return (
    <div className="flex items-start gap-3">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-gray-100 to-gray-200 text-gray-600 flex items-center justify-center shadow-sm">
        <Bot className="h-4 w-4" />
      </div>
      <div className="flex-1 max-w-3xl">
        <ToolCallDisplay toolCalls={message.tool_calls || []} forceRefresh={forceRefresh} />

        <div className="bg-white text-gray-800 p-4 rounded-2xl rounded-tl-none shadow-lg border border-gray-100 overflow-x-auto">
          <div className="prose prose-xs prose-gray max-w-none bg-transparent">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-1">{new Date(message.createdAt).toLocaleTimeString()}</p>

        {chartToolCall && (
          <div className="mt-3">
            <Button
              onClick={() => {
                let chartData = null;
                if (typeof chartToolCall.output === "string") {
                  try {
                    chartData = JSON.parse(chartToolCall.output);
                  } catch {
                    return;
                  }
                } else {
                  chartData = chartToolCall.output;
                }
                onOpenChart(chartData);
              }}
              size="sm"
              className="bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white shadow-md hover:shadow-lg transition-all duration-200"
            >
              <BarChart3 className="h-4 w-4 mr-2" />
              Open Canvas
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
