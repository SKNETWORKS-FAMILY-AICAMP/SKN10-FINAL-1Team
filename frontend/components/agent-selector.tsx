"use client"

import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { CodeIcon, FileTextIcon, BarChart3Icon, BotIcon, ChevronUpIcon, BrainIcon } from "lucide-react" // Added BrainIcon for prediction

// Map Django AgentType enum values to display names and icons
const agentTypeMap = {
  code: {
    icon: <CodeIcon className="h-4 w-4 text-blue-500" />,
    name: "Code Analysis Agent",
    description: "Code search, generation, and analysis",
  },
  rag: {
    icon: <FileTextIcon className="h-4 w-4 text-purple-500" />,
    name: "Document QA Agent",
    description: "Internal document search and analysis",
  },
  analytics: {
    icon: <BarChart3Icon className="h-4 w-4 text-green-500" />,
    name: "Business Analysis Agent",
    description: "Data visualization, SQL queries, and general business analysis",
  },
  prediction: { // New Prediction Agent
    icon: <BrainIcon className="h-4 w-4 text-purple-500" />,
    name: "Data Prediction Agent",
    description: "Performs machine learning predictions (e.g., customer churn) using CSV data.",
  },
  customer_ml_agent: {
    icon: <BrainIcon className="h-4 w-4 text-teal-500" />, // Using BrainIcon with a different color for now
    name: "Customer CSV Agent",
    description: "Answers questions about uploaded customer CSV data using a dedicated ML model.",
  },
  auto: {
    icon: <BotIcon className="h-4 w-4 text-slate-500" />,
    name: "Auto-detect Agent",
    description: "Automatically route to the best agent",
  },
}

export type AgentTypeKey = keyof typeof agentTypeMap;

export interface AgentSelectorProps {
  selectedAgent: AgentTypeKey;
  onAgentChange: (value: AgentTypeKey) => void;
}

export function AgentSelector({ selectedAgent, onAgentChange }: AgentSelectorProps) {
  return (
    <div className="border-t p-2 bg-slate-50 dark:bg-slate-900">
      <Popover>
        <PopoverTrigger asChild>
          <Button variant="outline" size="sm" className="w-full justify-between">
            <div className="flex items-center gap-2">
              {agentTypeMap[selectedAgent].icon}
              <span>{agentTypeMap[selectedAgent].name}</span>
            </div>
            <ChevronUpIcon className="h-4 w-4 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-80" align="center">
          <div className="space-y-4">
            <h4 className="font-medium">Select Agent</h4>
            <RadioGroup value={selectedAgent} onValueChange={(value) => onAgentChange(value as AgentTypeKey)} className="gap-3">
              <div className="flex items-center space-x-2 border rounded-md p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900">
                <RadioGroupItem value="auto" id="auto" />
                <Label htmlFor="auto" className="flex items-center gap-2 cursor-pointer">
                  {agentTypeMap.auto.icon}
                  <div>
                    <div>{agentTypeMap.auto.name}</div>
                    <div className="text-xs text-slate-500">{agentTypeMap.auto.description}</div>
                  </div>
                </Label>
              </div>

              <div className="flex items-center space-x-2 border rounded-md p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900">
                <RadioGroupItem value="code" id="code" />
                <Label htmlFor="code" className="flex items-center gap-2 cursor-pointer">
                  {agentTypeMap.code.icon}
                  <div>
                    <div>{agentTypeMap.code.name}</div>
                    <div className="text-xs text-slate-500">{agentTypeMap.code.description}</div>
                  </div>
                </Label>
              </div>

              <div className="flex items-center space-x-2 border rounded-md p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900">
                <RadioGroupItem value="rag" id="rag" />
                <Label htmlFor="rag" className="flex items-center gap-2 cursor-pointer">
                  {agentTypeMap.rag.icon}
                  <div>
                    <div>{agentTypeMap.rag.name}</div>
                    <div className="text-xs text-slate-500">{agentTypeMap.rag.description}</div>
                  </div>
                </Label>
              </div>

              <div className="flex items-center space-x-2 border rounded-md p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900">
                <RadioGroupItem value="analytics" id="analytics" />
                <Label htmlFor="analytics" className="flex items-center gap-2 cursor-pointer">
                  {agentTypeMap.analytics.icon}
                  <div>
                    <div>{agentTypeMap.analytics.name}</div>
                    <div className="text-xs text-slate-500">{agentTypeMap.analytics.description}</div>
                  </div>
                </Label>
              </div>

              <div className="flex items-center space-x-2 border rounded-md p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900">
                <RadioGroupItem value="prediction" id="prediction" />
                <Label htmlFor="prediction" className="flex items-center gap-2 cursor-pointer">
                  {agentTypeMap.prediction.icon}
                  <div>
                    <div>{agentTypeMap.prediction.name}</div>
                    <div className="text-xs text-slate-500">{agentTypeMap.prediction.description}</div>
                  </div>
                </Label>
              </div>

              {/* New Customer ML Agent Option */}
              <div className="flex items-center space-x-2 border rounded-md p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900">
                <RadioGroupItem value="customer_ml_agent" id="customer_ml_agent" />
                <Label htmlFor="customer_ml_agent" className="flex items-center gap-2 cursor-pointer">
                  {agentTypeMap.customer_ml_agent.icon}
                  <div>
                    <div>{agentTypeMap.customer_ml_agent.name}</div>
                    <div className="text-xs text-slate-500">{agentTypeMap.customer_ml_agent.description}</div>
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
