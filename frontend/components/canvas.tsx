"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CodeBlock } from "@/components/code-block"
import { DocumentSummary } from "@/components/document-summary"
import { BusinessChart } from "@/components/business-chart"
import { XIcon, DownloadIcon, CopyIcon, CheckIcon } from "lucide-react"

export function Canvas({ content, onClose }) {
  const [activeTab, setActiveTab] = useState("preview")
  const [copied, setCopied] = useState(false)

  const copyToClipboard = () => {
    if (content?.type === "code" && content.content) {
      navigator.clipboard.writeText(content.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const downloadContent = () => {
    if (content?.type === "code" && content.content) {
      const element = document.createElement("a")
      const file = new Blob([content.content], { type: "text/plain" })
      element.href = URL.createObjectURL(file)
      element.download = `code.${content.language === "javascript" ? "js" : content.language || "txt"}`
      document.body.appendChild(element)
      element.click()
      document.body.removeChild(element)
    }
  }

  return (
    <div className="h-full flex flex-col">
      <div className="border-b p-3 flex items-center justify-between bg-slate-50 dark:bg-slate-800">
        <h3 className="font-medium">{content?.title || "Canvas"}</h3>
        <div className="flex items-center gap-2">
          {content?.type === "code" && (
            <>
              <Button variant="ghost" size="icon" onClick={copyToClipboard}>
                {copied ? <CheckIcon className="h-4 w-4" /> : <CopyIcon className="h-4 w-4" />}
                <span className="sr-only">Copy</span>
              </Button>
              <Button variant="ghost" size="icon" onClick={downloadContent}>
                <DownloadIcon className="h-4 w-4" />
                <span className="sr-only">Download</span>
              </Button>
            </>
          )}
          <Button variant="ghost" size="icon" onClick={onClose}>
            <XIcon className="h-4 w-4" />
            <span className="sr-only">Close</span>
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <div className="border-b px-3">
          <TabsList className="h-10">
            <TabsTrigger value="preview">Preview</TabsTrigger>
            {content?.type === "code" && <TabsTrigger value="code">Code</TabsTrigger>}
            {content?.type === "document" && <TabsTrigger value="details">Details</TabsTrigger>}
            {content?.type === "chart" && <TabsTrigger value="data">Data</TabsTrigger>}
          </TabsList>
        </div>

        <TabsContent value="preview" className="flex-1 p-4 overflow-auto">
          {content?.type === "code" && <CodeBlock code={content.content} language={content.language} />}
          {content?.type === "document" && content.data && <DocumentSummary data={content.data} />}
          {content?.type === "chart" && content.data && <BusinessChart data={content.data} />}
        </TabsContent>

        <TabsContent value="code" className="flex-1 p-4 overflow-auto">
          {content?.content && (
            <pre className="bg-slate-100 dark:bg-slate-800 p-4 rounded-md overflow-x-auto">
              <code>{content.content}</code>
            </pre>
          )}
        </TabsContent>

        <TabsContent value="details" className="flex-1 p-4 overflow-auto">
          {content?.data && (
            <Card>
              <CardHeader>
                <CardTitle>Document Metadata</CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="space-y-2">
                  <div>
                    <dt className="font-medium">Title</dt>
                    <dd>{content.data.title || "Untitled"}</dd>
                  </div>
                  <div>
                    <dt className="font-medium">Type</dt>
                    <dd>{content.data.type || "Unknown"}</dd>
                  </div>
                  <div>
                    <dt className="font-medium">Date</dt>
                    <dd>{content.data.date || "No date"}</dd>
                  </div>
                  <div>
                    <dt className="font-medium">Source</dt>
                    <dd>{content.data.source || "Unknown source"}</dd>
                  </div>
                </dl>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="data" className="flex-1 p-4 overflow-auto">
          {content?.data?.chartData && (
            <Card>
              <CardHeader>
                <CardTitle>Chart Data</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="bg-slate-100 dark:bg-slate-800 p-4 rounded-md overflow-x-auto">
                  <code>{JSON.stringify(content.data.chartData, null, 2)}</code>
                </pre>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
