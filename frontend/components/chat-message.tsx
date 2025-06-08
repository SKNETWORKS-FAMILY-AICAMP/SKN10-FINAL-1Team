"use client"

import { useState } from "react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { CodeIcon, FileTextIcon, BarChart3Icon, ThumbsUpIcon, ThumbsDownIcon, CopyIcon, CheckIcon } from "lucide-react"
import { cleanFullMessageContent } from "@/lib/api/token-cleaner"
import ReactMarkdown from "react-markdown"
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { nord } from "react-syntax-highlighter/dist/cjs/styles/prism"

// Define TypeScript interface for the message prop
interface ChatMessageProps {
  message: {
    role: string;
    content: string;
    agentType?: string;
    id?: string;
  };
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [copied, setCopied] = useState(false)
  const [feedback, setFeedback] = useState<string | null>(null)

  // Clean the message content
  const cleanedContent = message.role === 'user' ? 
    message.content : 
    cleanFullMessageContent(message.content)

  const copyToClipboard = () => {
    navigator.clipboard.writeText(cleanedContent)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const giveFeedback = (value: string) => {
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
          <div className="bg-primary text-primary-foreground rounded-lg px-4 py-2">
            {/* User messages can also use markdown rendering */}
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // Simple styling for user messages
                code(props) {
                  const {children, className, ...rest} = props
                  return (
                    <code className={`${className} bg-primary-foreground/20 rounded px-1`} {...rest}>
                      {children}
                    </code>
                  )
                },
                // Other components can be customized here
                p: ({children}) => <p className="mb-1">{children}</p>,
                ul: ({children}) => <ul className="list-disc pl-4 mb-2">{children}</ul>,
                ol: ({children}) => <ol className="list-decimal pl-4 mb-2">{children}</ol>,
              }}
            >
              {cleanedContent}
            </ReactMarkdown>
          </div>
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
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code(props) {
                      const {children, className, node, ...rest} = props
                      const match = /language-(\w+)/.exec(className || '')
                      return match ? (
                        <div className="my-4 rounded-md overflow-hidden">
                          <div className="bg-slate-800 px-4 py-1 text-xs text-slate-300 flex justify-between items-center">
                            <span>{match[1]}</span>
                            <button 
                              onClick={() => navigator.clipboard.writeText(String(children))} 
                              className="text-slate-300 hover:text-white">
                              <CopyIcon className="h-3.5 w-3.5" />
                            </button>
                          </div>
                          <SyntaxHighlighter
                            // @ts-ignore - Type issues with the style property
                            style={nord}
                            language={match[1]}
                            PreTag="div"
                            customStyle={{ margin: 0, borderRadius: 0 }}
                            {...rest}
                          >
                            {String(children).replace(/\n$/, '')}
                          </SyntaxHighlighter>
                        </div>
                      ) : (
                        <code className={`${className || ''} bg-slate-100 dark:bg-slate-800 rounded px-1`} {...rest}>
                          {children}
                        </code>
                      )
                    },
                    h1: ({children}) => <h1 className="text-xl font-bold mt-4 mb-2">{children}</h1>,
                    h2: ({children}) => <h2 className="text-lg font-bold mt-4 mb-1">{children}</h2>,
                    h3: ({children}) => <h3 className="text-md font-bold mt-3 mb-1">{children}</h3>,
                    p: ({children}) => <p className="mb-4">{children}</p>,
                    ul: ({children}) => <ul className="list-disc pl-5 mb-4">{children}</ul>,
                    ol: ({children}) => <ol className="list-decimal pl-5 mb-4">{children}</ol>,
                    blockquote: ({children}) => <blockquote className="border-l-4 border-slate-300 dark:border-slate-600 pl-4 italic my-4">{children}</blockquote>,
                    table: ({children}) => <div className="overflow-x-auto my-4"><table className="border-collapse w-full border-spacing-0 border border-slate-300 dark:border-slate-600">{children}</table></div>,
                    thead: ({children}) => <thead className="bg-slate-50 dark:bg-slate-800">{children}</thead>,
                    tbody: ({children}) => <tbody>{children}</tbody>,
                    tr: ({children}) => <tr>{children}</tr>,
                    th: ({children}) => <th className="border border-slate-300 dark:border-slate-600 px-4 py-2 text-left font-semibold">{children}</th>,
                    td: ({children}) => <td className="border border-slate-300 dark:border-slate-600 px-4 py-2">{children}</td>,
                  }}
                >
                  {cleanedContent}
                </ReactMarkdown>

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
