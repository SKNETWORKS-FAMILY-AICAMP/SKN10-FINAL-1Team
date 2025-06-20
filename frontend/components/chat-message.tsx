"use client"

import { useState, useEffect, useRef, memo } from "react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { CodeIcon, FileTextIcon, BarChart3Icon, ThumbsUpIcon, ThumbsDownIcon, CopyIcon, CheckIcon } from "lucide-react"
import { cleanFullMessageContent } from "@/lib/api/token-cleaner"
import ReactMarkdown from "react-markdown"
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { nord } from "react-syntax-highlighter/dist/cjs/styles/prism"
import mermaid from "mermaid"

// Define TypeScript interface for the message prop
interface ChatMessageProps {
  message: {
    role: string;
    content: string;
    agentType?: string;
    id?: string;
  };
  isStreaming?: boolean; // 스트리밍 상태를 나타내는 prop 추가
}

// 머메이드 다이어그램 감지 함수
const isMermaidDiagram = (content: string): boolean => {
  if (!content) return false;
  
  // 오직 머메이드 코드 블록으로 감싸진 경우만 감지 (코드 블록 밖은 무시)
  return content.includes('```mermaid');
};

// 머메이드 다이어그램 추출 함수
const extractMermaidCode = (content: string): string => {
  // 마크다운 코드 블록에서만 추출
  const codeBlockMatch = content.match(/```mermaid([\s\S]*?)```/);
  if (codeBlockMatch && codeBlockMatch[1]) {
    return codeBlockMatch[1].trim();
  }
  
  // 코드 블록을 찾지 못한 경우 빈 문자열 반환
  return '';
};

// 디버깅용 로그 컴포넌트
const DebugLog = ({ show, data }: { show: boolean, data: any }) => {
  if (!show) return null;
  return (
    <div className="bg-red-100 text-red-800 p-2 text-xs rounded mt-2 overflow-auto max-h-40">
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
};

// Memoized Mermaid component to avoid re-rendering during streaming
const MermaidDiagram = memo(({ chart }: { chart: string }) => {
  const [currentSvg, setCurrentSvg] = useState<string>('');
  const [currentError, setCurrentError] = useState<string | null>(null);
  const lastSuccessfullyRenderedChart = useRef<string | null>(null);
  const diagramId = useRef<string>(`mermaid-${Math.random().toString(36).substring(2, 10)}`);
  const containerRef = useRef<HTMLDivElement>(null); // For potential future use with mermaid API if needed

  const codeToCopy = lastSuccessfullyRenderedChart.current || chart?.trim() || '';

  const copyChartCode = () => {
    if (codeToCopy) {
      navigator.clipboard.writeText(codeToCopy)
        .then(() => {
          // Optional: Show some feedback to the user
        })
        .catch(err => {
          console.error('Failed to copy chart code: ', err);
        });
    } else {
      console.warn('No chart code available to copy.');
    }
  };

  useEffect(() => {
    const newChartPropTrimmed = chart?.trim();

    if (newChartPropTrimmed) {
      if (newChartPropTrimmed !== lastSuccessfullyRenderedChart.current) {
        mermaid.render(diagramId.current, newChartPropTrimmed)
          .then(({ svg: newSvgString }) => {
            // More robust check for Mermaid's own error SVG renderings
            const isMermaidErrorOutput = 
              newSvgString.includes('aria-roledescription="error"') || // Primary check for error SVGs
              newSvgString.includes('class="error-text"') || 
              newSvgString.includes("Syntax error in text") || 
              newSvgString.includes("mermaidAPI.render error");

            if (isMermaidErrorOutput) {
              console.error("Mermaid generated an error SVG. Chart:", newChartPropTrimmed);
              // Only show error in UI if no previous successful render exists for this instance.
              if (!lastSuccessfullyRenderedChart.current) {
                setCurrentError("Mermaid 다이어그램 처리 중 오류가 발생했습니다.");
                setCurrentSvg('');
              }
              // If a previous diagram was fine, keep showing it and ignore transient error.
            } else {
              // Check if newSvgString is effectively empty
              const isEmptySvg = !newSvgString || newSvgString.trim() === '' || newSvgString.trim() === '<svg></svg>';

              if (isEmptySvg) {
                console.warn("Mermaid rendered an empty or near-empty SVG. Chart:", newChartPropTrimmed);
                // Only show error or clear if no previous successful render exists for this instance.
                if (!lastSuccessfullyRenderedChart.current) { 
                  setCurrentError("Mermaid 다이어그램이 비어있거나 내용을 생성하지 못했습니다.");
                  setCurrentSvg('');
                }
                // If lastSuccessfullyRenderedChart.current exists (from a previous good render in this instance),
                // do nothing, thereby keeping the old good SVG, making it stickier against this case.
              } else {
                // This is a good, non-empty, non-error SVG
                setCurrentSvg(newSvgString);
                setCurrentError(null);
                lastSuccessfullyRenderedChart.current = newChartPropTrimmed;
              }
            }
          })
          .catch((err: any) => {
            console.error("Mermaid rendering error (catch block):", err, "Chart:", newChartPropTrimmed);
            // Only show error in UI if no previous successful render exists for this instance.
            if (!lastSuccessfullyRenderedChart.current) {
              setCurrentError(err?.message || "머메이드 다이어그램 렌더링 중 오류 발생");
              setCurrentSvg(''); 
            }
            // If a previous diagram was fine, keep showing it and ignore transient error from incomplete stream.
          });
      }
    } else {
      if (!lastSuccessfullyRenderedChart.current) {
        setCurrentSvg('');
        setCurrentError(null);
      }
      // If lastSuccessfullyRenderedChart.current exists, do nothing, keep currentSvg (sticky).
    }
  }, [chart]); // Only depend on the 'chart' prop.

  if (currentError) {
    return <div className="text-red-500 text-sm py-2">Error: {currentError}</div>;
  }

  if (currentSvg) {
    return (
      <div className="mermaid-diagram-container relative p-2 border rounded-md my-2 bg-white dark:bg-gray-800 overflow-x-auto">
        <div ref={containerRef} dangerouslySetInnerHTML={{ __html: currentSvg }} />
        {codeToCopy && (
          <button 
            onClick={copyChartCode}
            className="absolute top-2 right-2 p-1 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors opacity-50 hover:opacity-100"
            title="Copy Mermaid Code"
          >
            <CopyIcon className="h-4 w-4" />
          </button>
        )}
      </div>
    );
  }

  // If chart prop is empty AND no diagram was ever successfully rendered for this instance.
  if (!chart?.trim() && !lastSuccessfullyRenderedChart.current) {
    return <div className="text-yellow-500 text-sm py-2">머메이드 다이어그램이 비어 있습니다</div>;
  }
  
  // Fallback: if no SVG, no error, and it's not an explicitly empty initial chart 
  // (e.g. chart prop is valid but still rendering, or was valid and then prop became empty but we are sticky)
  // This state should ideally be covered by currentSvg having content if sticky and previously rendered.
  // If currentSvg is empty here, it means it's either loading (initial render of a valid chart) or truly empty.
  return null; 
});

MermaidDiagram.displayName = "MermaidDiagram"

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, isStreaming }) => {
  const [copied, setCopied] = useState(false)
  const [feedback, setFeedback] = useState<string | null>(null)

  // Clean the message content
  const cleanedContent = message.role === 'user' ? 
    message.content : 
    cleanFullMessageContent(message.content)
    
  // 머메이드 다이어그램 코드는 이제 ReactMarkdown 내부에서만 처리

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
            
            {/* 머메이드 다이어그램은 ReactMarkdown 내부에서만 렌더링됨 */}

            <div className="bg-white dark:bg-slate-900 rounded-lg px-4 py-3 shadow-sm border">
              <div className="prose dark:prose-invert prose-sm max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code(props) {
                      const {children, className, ...rest} = props
                      const match = /language-(\w+)/.exec(className || '')
                      const language = match ? match[1] : ''

                      // For inline code without language specification
                      if (!match) {
                        return (
                          <code className={`${className || ''} bg-slate-100 dark:bg-slate-800 rounded px-1`} {...rest}>
                            {children}
                          </code>
                        )
                      }

                      // Special handling for Mermaid diagrams
                      if (language?.toLowerCase() === 'mermaid') {
                        const mermaidCode = String(children).trim();
                        if (isStreaming) {
                          // 스트리밍 중에는 Mermaid 코드를 텍스트로 표시
                          return (
                            <div className="my-4 p-4 bg-slate-100 dark:bg-slate-800 rounded-md">
                              <p className="text-sm text-slate-500 dark:text-slate-400 mb-2">Mermaid 다이어그램 로딩 중...</p>
                              <pre className="text-xs whitespace-pre-wrap"><code>{mermaidCode}</code></pre>
                            </div>
                          );
                        }
                        // 스트리밍 완료 시 다이어그램 렌더링
                        return mermaidCode ? <MermaidDiagram chart={mermaidCode} /> : null;
                      }

                      return (
                        <div className="my-4 rounded-md overflow-hidden">
                          <div className="bg-slate-800 px-4 py-1 text-xs text-slate-300 flex justify-between items-center">
                            <span>{language}</span>
                            <button 
                              className="text-slate-300 hover:text-white"
                              onClick={() => navigator.clipboard.writeText(String(children).replace(/\n$/, ''))}
                            >
                              <CopyIcon className="h-3.5 w-3.5" />
                            </button>
                          </div>
                          <SyntaxHighlighter
                            language={language}
                            style={nord}
                            customStyle={{ margin: 0, borderRadius: 0 }}
                            showLineNumbers={language !== 'bash' && language !== 'shell'}
                          >
                            {String(children).replace(/\n$/, '')}
                          </SyntaxHighlighter>
                        </div>
                      )
                    },
                    h1: ({children}) => <h1 className="text-xl font-bold mt-4 mb-2">{children}</h1>,
                    h2: ({children}) => <h2 className="text-lg font-bold mt-4 mb-1">{children}</h2>,
                    h3: ({children}) => <h3 className="text-md font-bold mt-3 mb-1">{children}</h3>,
                    p: ({children}) => <p className="mb-4">{children}</p>,
                    ul: ({children}) => <ul className="list-disc pl-5 mb-4">{children}</ul>,
                    ol: ({children}) => <ol className="list-decimal pl-5 mb-4">{children}</ol>,
                    blockquote: ({children}) => <blockquote className="border-l-4 border-slate-300 dark:border-slate-600 pl-4 italic my-4">{children}</blockquote>,
                    table: ({children}) => <div className="overflow-x-auto my-4"><table className="border-collapse w-full border-spacing-0 border border-slate-300 dark:border-slate-600 mb-4">{children}</table></div>,
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
