export type TMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  createdAt: string
  tool_calls?: TToolCall[]
}

export type TToolCall = {
  id: string
  name: string
  args: Record<string, unknown>
  output?: string | object | null
}

export type TSession = {
  id: string
  title: string
  started_at: string
}

export type ChartContent = {
  canvas_html: string
  script_js: string
}

export interface StreamMessage {
  type: string
  tool_call_id?: string
  content?: string
}

export interface ToolMessage extends StreamMessage {
  type: "tool"
  tool_call_id: string
  content: string
}

export interface AIMessage extends StreamMessage {
  type: "ai"
  tool_calls?: TToolCall[]
}
