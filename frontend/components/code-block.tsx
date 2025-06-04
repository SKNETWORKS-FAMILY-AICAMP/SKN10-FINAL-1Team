"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { CopyIcon, CheckIcon } from "lucide-react"

export function CodeBlock({ code, language }) {
  const [copied, setCopied] = useState(false)

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="relative mt-4 rounded-md overflow-hidden">
      <div className="flex items-center justify-between bg-slate-800 px-4 py-2 text-xs text-slate-200">
        <span>{language || "code"}</span>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 text-slate-200 hover:text-white hover:bg-slate-700"
          onClick={copyToClipboard}
        >
          {copied ? <CheckIcon className="h-3.5 w-3.5" /> : <CopyIcon className="h-3.5 w-3.5" />}
          <span className="sr-only">Copy code</span>
        </Button>
      </div>
      <pre className="bg-slate-900 p-4 overflow-x-auto text-slate-100 text-sm">
        <code>{code}</code>
      </pre>
    </div>
  )
}
