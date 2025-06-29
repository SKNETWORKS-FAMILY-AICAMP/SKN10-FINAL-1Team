"use client"

import { Bot, BarChart3, Copy, Check } from "lucide-react"
import { Button } from "@/components/ui/button"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import type { TMessage, ChartContent } from "@/types/chat"
import { ToolCallDisplay } from "./ToolCallDisplay"
import { useMemo, useState, useEffect } from "react"

// Prism.js 타입 확장
declare global {
  interface Window {
    Prism?: {
      highlightAll: () => void;
    };
  }
}

interface ChatMessageProps {
  message: TMessage
  chartContent: ChartContent | null
  onOpenChart: (data?: any) => void
  forceRefresh?: boolean
}

export function ChatMessage({ message, chartContent, onOpenChart, forceRefresh = false }: ChatMessageProps) {
  const [copied, setCopied] = useState(false)

  // Prism.js 하이라이팅 적용
  useEffect(() => {
    if (typeof window !== 'undefined' && window.Prism) {
      window.Prism.highlightAll()
    }
  }, [message.content])

  const handleCopy = async () => {
    try {
      // 먼저 navigator.clipboard API 시도
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(message.content)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
        return
      }
      
      // fallback: document.execCommand 사용
      const textArea = document.createElement('textarea')
      textArea.value = message.content
      textArea.style.position = 'fixed'
      textArea.style.left = '-999999px'
      textArea.style.top = '-999999px'
      document.body.appendChild(textArea)
      textArea.focus()
      textArea.select()
      
      const successful = document.execCommand('copy')
      document.body.removeChild(textArea)
      
      if (successful) {
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      } else {
        console.error('Failed to copy text using fallback method')
      }
    } catch (err) {
      console.error('Failed to copy text: ', err)
    }
  }

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-xl">
          <div className="bg-gradient-to-r from-blue-600 to-blue-500 text-white p-4 rounded-2xl rounded-tr-none shadow-lg">
            <p className="text-sm">{message.content}</p>
          </div>
          <div className="flex items-center justify-between mt-1">
            <p className="text-xs text-gray-500">{new Date(message.createdAt).toLocaleTimeString()}</p>
            
            {/* 사용자 메시지 복사 버튼 */}
            {message.content && message.content.trim() && (
              <Button
                onClick={handleCopy}
                size="sm"
                variant="ghost"
                className="h-6 px-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-all duration-200 rounded-lg"
                title="Copy message content"
              >
                {copied ? (
                  <>
                    <Check className="h-3 w-3 mr-1" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3 mr-1" />
                    Copy
                  </>
                )}
              </Button>
            )}
          </div>
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
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                code({ node, className, children, ...props }: any) {
                  const match = /language-(\w+)/.exec(className || '')
                  const isInline = !match
                  return !isInline ? (
                    <pre className="code-editor-style">
                      <code className={className} {...props}>
                        {children}
                      </code>
                    </pre>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  )
                }
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        </div>
        
        <div className="flex items-center justify-between mt-2">
          <p className="text-xs text-gray-500">{new Date(message.createdAt).toLocaleTimeString()}</p>
          
          {/* 복사 버튼 - 메시지 내용이 있을 때만 표시 */}
          {message.content && message.content.trim() && (
            <Button
              onClick={handleCopy}
              size="sm"
              variant="ghost"
              className="h-8 px-3 text-gray-500 hover:text-gray-700 hover:bg-gray-100 transition-all duration-200 rounded-lg"
              title="Copy message content"
            >
              {copied ? (
                <>
                  <Check className="h-3 w-3 mr-1" />
                  Copied!
                </>
              ) : (
                <>
                  <Copy className="h-3 w-3 mr-1" />
                  Copy
                </>
              )}
            </Button>
          )}
        </div>

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
