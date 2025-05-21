"use client"

import { useState } from "react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { CodeIcon, FileTextIcon, BarChart3Icon, ThumbsUpIcon, ThumbsDownIcon, CopyIcon, CheckIcon } from "lucide-react"

export function ChatMessage({ message }) {
  const [copied, setCopied] = useState(false)
  const [feedback, setFeedback] = useState(null)

  const copyToClipboard = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const giveFeedback = (value) => {
    setFeedback(value)
    // In a real app, this would send feedback to the server
  }

  return (
    <div className={`flex items-start gap-3 ${message.role === "user" ? "justify-end" : ""}`}>
      {message.role !== "user" && (
        <Avatar className="h-8 w-8">
          <AvatarImage src="/placeholder.svg?height=32&width=32" />
          <AvatarFallback>AI</AvatarFallback>
        </Avatar>
      )}

      <div className={`max-w-[80%] ${message.role === "user" ? "order-1" : "order-2"}`}>
        {message.role === "user" ? (
          <div className="bg-primary text-primary-foreground rounded-lg px-4 py-2">{message.content}</div>
        ) : (
          <div className="space-y-2">
            {message.agentType && (
              <div className="flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400 mb-1">
                {message.agentType === "code" && <CodeIcon className="h-3 w-3" />}
                {message.agentType === "document" && <FileTextIcon className="h-3 w-3" />}
                {message.agentType === "business" && <BarChart3Icon className="h-3 w-3" />}
                <span>
                  {message.agentType === "code" && "Code Analysis Agent"}
                  {message.agentType === "document" && "Document QA Agent"}
                  {message.agentType === "business" && "Business Analysis Agent"}
                </span>
              </div>
            )}

            <div className="bg-white dark:bg-slate-900 rounded-lg px-4 py-3 shadow-sm border">
              <div className="prose dark:prose-invert prose-sm max-w-none">
                {message.content}

                {/* Note: We no longer render code/charts/documents here as they're shown in the Canvas */}
                {message.agentType && (
                  <div className="mt-2 text-xs text-slate-500 dark:text-slate-400 italic">
                    {message.agentType === "code" && "Code is displayed in the canvas panel →"}
                    {message.agentType === "document" && "Document details are displayed in the canvas panel →"}
                    {message.agentType === "business" && "Chart is displayed in the canvas panel →"}
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2 justify-end">
              <Button variant="ghost" size="icon" className="h-7 w-7" onClick={copyToClipboard}>
                {copied ? <CheckIcon className="h-4 w-4" /> : <CopyIcon className="h-4 w-4" />}
                <span className="sr-only">Copy</span>
              </Button>

              <Button
                variant="ghost"
                size="icon"
                className={`h-7 w-7 ${feedback === "up" ? "text-green-500" : ""}`}
                onClick={() => giveFeedback("up")}
              >
                <ThumbsUpIcon className="h-4 w-4" />
                <span className="sr-only">Helpful</span>
              </Button>

              <Button
                variant="ghost"
                size="icon"
                className={`h-7 w-7 ${feedback === "down" ? "text-red-500" : ""}`}
                onClick={() => giveFeedback("down")}
              >
                <ThumbsDownIcon className="h-4 w-4" />
                <span className="sr-only">Not helpful</span>
              </Button>
            </div>
          </div>
        )}
      </div>

      {message.role === "user" && (
        <Avatar className="h-8 w-8 order-2">
          <AvatarImage src="/placeholder.svg?height=32&width=32" />
          <AvatarFallback>U</AvatarFallback>
        </Avatar>
      )}
    </div>
  )
}
