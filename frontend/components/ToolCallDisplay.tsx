"use client"

import { useState, useEffect, useRef } from "react"
import {
  Settings,
  ChevronDown,
  ChevronRight,
  Code,
  Database,
  BarChart3,
  FileText,
  Zap,
  CheckCircle,
  Clock,
} from "lucide-react"
import type { TToolCall } from "@/types/chat"

interface ToolCallDisplayProps {
  toolCalls: TToolCall[]
  forceRefresh?: boolean // 강제 리프레시 플래그 추가
}

// 도구별 아이콘 매핑
const getToolIcon = (toolName: string) => {
  const name = toolName.toLowerCase()
  if (name.includes("chart") || name.includes("graph")) return BarChart3
  if (name.includes("database") || name.includes("sql")) return Database
  if (name.includes("code") || name.includes("python")) return Code
  if (name.includes("file") || name.includes("document")) return FileText
  return Zap
}

// 도구별 색상 테마
const getToolTheme = (toolName: string) => {
  const name = toolName.toLowerCase()
  if (name.includes("chart") || name.includes("graph")) {
    return {
      bg: "bg-gradient-to-r from-purple-50 to-pink-50",
      border: "border-purple-200",
      icon: "text-purple-600",
      text: "text-purple-700",
      accent: "bg-purple-100",
    }
  }
  if (name.includes("database") || name.includes("sql")) {
    return {
      bg: "bg-gradient-to-r from-green-50 to-emerald-50",
      border: "border-green-200",
      icon: "text-green-600",
      text: "text-green-700",
      accent: "bg-green-100",
    }
  }
  if (name.includes("code") || name.includes("python")) {
    return {
      bg: "bg-gradient-to-r from-blue-50 to-indigo-50",
      border: "border-blue-200",
      icon: "text-blue-600",
      text: "text-blue-700",
      accent: "bg-blue-100",
    }
  }
  return {
    bg: "bg-gradient-to-r from-orange-50 to-amber-50",
    border: "border-orange-200",
    icon: "text-orange-600",
    text: "text-orange-700",
    accent: "bg-orange-100",
  }
}

export function ToolCallDisplay({ toolCalls, forceRefresh = false }: ToolCallDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(true) // 기본적으로 확장
  const [refreshKey, setRefreshKey] = useState(0) // 리프레시 키 추가
  const prevForceRefresh = useRef(forceRefresh)

  // forceRefresh가 변경되면 컴포넌트 강제 리프레시
  useEffect(() => {
    if (forceRefresh && !prevForceRefresh.current) {
      console.log("🔄 ToolCallDisplay 강제 리프레시 실행")
      setRefreshKey(prev => prev + 1)
      
      // 강제 리프레시 후 도구 상태 재확인
      setTimeout(() => {
        const completedCount = toolCalls.filter(call => call.output).length
        const totalCount = toolCalls.length
        console.log(`🔍 강제 리프레시 후 도구 상태: ${completedCount}/${totalCount} 완료`)
      }, 100)
    }
    prevForceRefresh.current = forceRefresh
  }, [forceRefresh, toolCalls])

  // 도구 호출 상태가 변경될 때마다 디버깅 로그
  useEffect(() => {
    const completedCount = toolCalls.filter(call => call.output).length
    const totalCount = toolCalls.length
    console.log(`🔧 ToolCallDisplay 상태 업데이트: ${completedCount}/${totalCount} 완료`)
    
    // 모든 도구가 완료되었는지 확인
    if (totalCount > 0 && completedCount === totalCount) {
      console.log("✅ 모든 도구 호출이 완료되었습니다!")
    } else if (totalCount > 0 && completedCount > 0) {
      console.log(`⏳ 진행 중: ${completedCount}/${totalCount} 완료`)
    }
  }, [toolCalls])

  if (!toolCalls || toolCalls.length === 0) return null

  const allToolsFinished = toolCalls.every((call) => call.output)
  const runningTools = toolCalls.filter((call) => !call.output).length
  const completedTools = toolCalls.length - runningTools

  return (
    <div key={refreshKey} className="mb-4 rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      {/* 헤더 */}
      <div
        className="px-4 py-3 bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-blue-100 cursor-pointer hover:from-blue-100 hover:to-indigo-100 transition-all duration-200"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Settings className="h-4 w-4 text-blue-600" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-blue-800">AI 도구 실행</h4>
              <p className="text-xs text-blue-600">
                {completedTools}개 완료, {runningTools}개 실행 중
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1">
              {allToolsFinished ? (
                <CheckCircle className="h-4 w-4 text-green-500" />
              ) : (
                <Clock className="h-4 w-4 text-blue-500 animate-spin" />
              )}
              <span
                className={`text-xs font-medium ${
                  allToolsFinished ? "text-green-600" : "text-blue-600"
                }`}
              >
                {allToolsFinished ? "완료" : "실행 중..."}
              </span>
            </div>
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 text-blue-600 transition-transform duration-200" />
            ) : (
              <ChevronRight className="h-4 w-4 text-blue-600 transition-transform duration-200" />
            )}
          </div>
        </div>
      </div>

      {/* 확장된 내용 */}
      <div
        className={`transition-all duration-300 ease-in-out ${
          isExpanded ? "max-h-[500px] opacity-100" : "max-h-0 opacity-0"
        } overflow-hidden`}
      >
        <div className="p-2 space-y-2 bg-gray-50/50">
          <div className="space-y-2 max-h-96 overflow-y-auto p-2">
            {toolCalls.map((call, index) => {
              const IconComponent = getToolIcon(call.name)
              const theme = getToolTheme(call.name)
              const isFinished = !!call.output

              // 더 고유한 key 생성 (tool_call_id + index + refreshKey)
              const uniqueKey = `${call.id || call.name}-${index}-${refreshKey}`

              return (
                <div key={uniqueKey} className={`rounded-lg border ${theme.border} ${theme.bg} overflow-hidden`}>
                  {/* 도구 정보 */}
                  <div className="px-4 py-3 border-b border-gray-100">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 ${theme.accent} rounded-lg`}>
                        <IconComponent className={`h-4 w-4 ${theme.icon}`} />
                      </div>
                      <div className="flex-1">
                        <h5 className={`text-sm font-semibold ${theme.text}`}>{call.name}</h5>
                        <div className="flex items-center space-x-2 mt-1">
                          {isFinished ? (
                            <CheckCircle className="h-3 w-3 text-green-500" />
                          ) : (
                            <Clock className="h-3 w-3 text-gray-400 animate-spin" />
                          )}
                          <span className={`text-xs ${isFinished ? "text-gray-500" : "text-blue-500"}`}>
                            {isFinished ? "실행 완료" : "실행 중..."}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* 입력 파라미터 */}
                  <div className="px-4 py-3 bg-white/50">
                    <div className="mb-2">
                      <span className="text-xs font-medium text-gray-600 uppercase tracking-wide">입력 파라미터</span>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3 border border-gray-200 max-h-64 overflow-y-auto">
                      <pre className="text-xs text-gray-700 overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed">
                        {JSON.stringify(call.args, null, 2)}
                      </pre>
                    </div>
                  </div>

                  {/* 출력 결과 */}
                  {isFinished && (
                    <div className="px-4 py-3 bg-white/30 border-t border-gray-100">
                      <div className="mb-2 flex items-center space-x-2">
                        <CheckCircle className="h-3 w-3 text-green-500" />
                        <span className="text-xs font-medium text-gray-600 uppercase tracking-wide">실행 결과</span>
                      </div>
                      <div className="bg-green-50 rounded-lg p-3 border border-green-200 max-h-64 overflow-y-auto">
                        <pre className="text-xs text-green-800 overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed">
                          {typeof call.output === "string" ? call.output : JSON.stringify(call.output, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
