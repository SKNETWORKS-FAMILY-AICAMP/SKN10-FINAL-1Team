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
  args: any
  output?: any
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
