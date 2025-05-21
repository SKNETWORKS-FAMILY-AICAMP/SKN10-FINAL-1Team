"use client"

import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { CodeIcon, FileTextIcon, BarChart3Icon, BotIcon, ChevronUpIcon } from "lucide-react"

export function AgentSelector({ selectedAgent, onAgentChange }) {
  return (
    <div className="border-t p-2 bg-slate-50 dark:bg-slate-900">
      <Popover>
        <PopoverTrigger asChild>
          <Button variant="outline" size="sm" className="w-full justify-between">
            <div className="flex items-center gap-2">
              {selectedAgent === "auto" && <BotIcon className="h-4 w-4 text-slate-500" />}
              {selectedAgent === "code" && <CodeIcon className="h-4 w-4 text-blue-500" />}
              {selectedAgent === "document" && <FileTextIcon className="h-4 w-4 text-purple-500" />}
              {selectedAgent === "business" && <BarChart3Icon className="h-4 w-4 text-green-500" />}

              <span>
                {selectedAgent === "auto" && "Auto-detect Agent"}
                {selectedAgent === "code" && "Code Analysis Agent"}
                {selectedAgent === "document" && "Document QA Agent"}
                {selectedAgent === "business" && "Business Analysis Agent"}
              </span>
            </div>
            <ChevronUpIcon className="h-4 w-4 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-80" align="center">
          <div className="space-y-4">
            <h4 className="font-medium">Select Agent</h4>
            <RadioGroup value={selectedAgent} onValueChange={onAgentChange} className="gap-3">
              <div className="flex items-center space-x-2 border rounded-md p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900">
                <RadioGroupItem value="auto" id="auto" />
                <Label htmlFor="auto" className="flex items-center gap-2 cursor-pointer">
                  <BotIcon className="h-5 w-5 text-slate-500" />
                  <div>
                    <div>Auto-detect Agent</div>
                    <div className="text-xs text-slate-500">Automatically route to the best agent</div>
                  </div>
                </Label>
              </div>

              <div className="flex items-center space-x-2 border rounded-md p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900">
                <RadioGroupItem value="code" id="code" />
                <Label htmlFor="code" className="flex items-center gap-2 cursor-pointer">
                  <CodeIcon className="h-5 w-5 text-blue-500" />
                  <div>
                    <div>Code Analysis Agent</div>
                    <div className="text-xs text-slate-500">Code search, generation, and analysis</div>
                  </div>
                </Label>
              </div>

              <div className="flex items-center space-x-2 border rounded-md p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900">
                <RadioGroupItem value="document" id="document" />
                <Label htmlFor="document" className="flex items-center gap-2 cursor-pointer">
                  <FileTextIcon className="h-5 w-5 text-purple-500" />
                  <div>
                    <div>Document QA Agent</div>
                    <div className="text-xs text-slate-500">Internal document search and analysis</div>
                  </div>
                </Label>
              </div>

              <div className="flex items-center space-x-2 border rounded-md p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900">
                <RadioGroupItem value="business" id="business" />
                <Label htmlFor="business" className="flex items-center gap-2 cursor-pointer">
                  <BarChart3Icon className="h-5 w-5 text-green-500" />
                  <div>
                    <div>Business Analysis Agent</div>
                    <div className="text-xs text-slate-500">Data visualization and business insights</div>
                  </div>
                </Label>
              </div>
            </RadioGroup>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  )
}
