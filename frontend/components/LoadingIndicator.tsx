import { Bot } from "lucide-react"

export function LoadingIndicator() {
  return (
    <div className="flex items-center space-x-2">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 text-gray-600 flex items-center justify-center">
        <Bot className="h-4 w-4" />
      </div>
      <div className="flex space-x-1">
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
      </div>
      <span className="text-sm text-gray-500">생성 중...</span>
    </div>
  )
}
