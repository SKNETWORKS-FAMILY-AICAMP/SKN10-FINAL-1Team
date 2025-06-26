"use client"

import { useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { X } from "lucide-react"
import type { ChartContent } from "@/types/chat"

interface ChartModalProps {
  isOpen: boolean
  onClose: () => void
  chartContent: ChartContent | null
}

export function ChartModal({ isOpen, onClose, chartContent }: ChartModalProps) {
  const chartModalContentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isOpen && chartContent && chartModalContentRef.current) {
      chartModalContentRef.current.innerHTML = chartContent.canvas_html
      const script = document.createElement("script")
      script.textContent = `(function() { ${chartContent.script_js} })()`
      chartModalContentRef.current.appendChild(script)
    }
  }, [isOpen, chartContent])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg w-full max-w-4xl h-3/4 flex flex-col">
        <div className="flex justify-between items-center p-4 border-b border-gray-200">
          <h3 className="text-xl font-bold text-gray-800">차트 보기</h3>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div ref={chartModalContentRef} className="flex-1 overflow-y-auto p-4"></div>
      </div>
    </div>
  )
}
