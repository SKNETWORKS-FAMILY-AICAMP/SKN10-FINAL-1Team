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

        <div className="bg-white text-gray-800 p-4 rounded-2xl rounded-tl-none shadow-lg border border-gray-100">
          <div className="text-sm leading-relaxed prose prose-sm prose-gray max-w-none">
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                // 테이블 스타일링
                table: ({ children }) => (
                  <div className="overflow-x-auto my-4">
                    <table className="min-w-full border border-gray-200 rounded-lg">
                      {children}
                    </table>
                  </div>
                ),
                thead: ({ children }) => (
                  <thead className="bg-gray-50">{children}</thead>
                ),
                th: ({ children }) => (
                  <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700 border-b border-gray-200">
                    {children}
                  </th>
                ),
                td: ({ children }) => (
                  <td className="px-4 py-2 text-sm text-gray-600 border-b border-gray-100">
                    {children}
                  </td>
                ),
                // 코드 블록 스타일링
                code: ({ className, children, ...props }: any) => {
                  const match = /language-(\w+)/.exec(className || '')
                  const isInline = !match
                  return !isInline ? (
                    <pre className="bg-gray-100 p-3 rounded-lg overflow-x-auto my-4">
                      <code className={className} {...props}>
                        {children}
                      </code>
                    </pre>
                  ) : (
                    <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono" {...props}>
                      {children}
                    </code>
                  )
                },
                // 인용구 스타일링
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-blue-500 pl-4 my-4 italic text-gray-600">
                    {children}
                  </blockquote>
                ),
                // 링크 스타일링
                a: ({ children, href }) => (
                  <a href={href} className="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
                // 리스트 스타일링
                ul: ({ children }) => (
                  <ul className="list-disc list-inside my-4 space-y-1">{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal list-inside my-4 space-y-1">{children}</ol>
                ),
                // 제목 스타일링
                h1: ({ children }) => (
                  <h1 className="text-2xl font-bold text-gray-800 mt-6 mb-4">{children}</h1>
                ),
                h2: ({ children }) => (
                  <h2 className="text-xl font-bold text-gray-800 mt-5 mb-3">{children}</h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-lg font-bold text-gray-800 mt-4 mb-2">{children}</h3>
                ),
                h4: ({ children }) => (
                  <h4 className="text-base font-bold text-gray-800 mt-3 mb-2">{children}</h4>
                ),
                // 단락 스타일링
                p: ({ children }) => (
                  <p className="mb-3 last:mb-0">{children}</p>
                ),
              }}
            >
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
