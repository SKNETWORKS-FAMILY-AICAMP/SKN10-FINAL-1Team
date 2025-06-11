import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import Link from "next/link"
import { BarChart3Icon, FileTextIcon, CodeIcon, LogInIcon } from "lucide-react"

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl mb-6">AI Agent Platform</h1>
          <p className="text-xl text-slate-600 dark:text-slate-400 mb-8">
            Intelligent assistance for code analysis, document QA, and business insights
          </p>
          <div className="flex justify-center gap-4">
            <Button asChild size="lg" className="gap-2">
              <Link href="/login">
                <LogInIcon className="h-5 w-5" />
                Company Login
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="/demo">Try Demo</Link>
            </Button>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <Card className="border-t-4 border-t-blue-500">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CodeIcon className="h-5 w-5 text-blue-500" />
                Code Analysis Agent
              </CardTitle>
              <CardDescription>Analyze, search, and generate code with intelligent assistance</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc list-inside text-sm space-y-2 text-slate-600 dark:text-slate-400">
                <li>Repository document search</li>
                <li>Code location tracking</li>
                <li>Code generation and version conversion</li>
                <li>Related document auto-linking</li>
                <li>Analysis history visualization</li>
              </ul>
            </CardContent>
          </Card>

          <Card className="border-t-4 border-t-purple-500">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileTextIcon className="h-5 w-5 text-purple-500" />
                Document QA Agent
              </CardTitle>
              <CardDescription>Search and analyze internal documents with natural language</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc list-inside text-sm space-y-2 text-slate-600 dark:text-slate-400">
                <li>Document type classification</li>
                <li>Document summarization</li>
                <li>Policy/guideline search</li>
                <li>Source citation</li>
                <li>Document comparison</li>
              </ul>
            </CardContent>
          </Card>

          <Card className="border-t-4 border-t-green-500">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3Icon className="h-5 w-5 text-green-500" />
                Business Analysis Agent
              </CardTitle>
              <CardDescription>Data-driven insights and visualizations for business decisions</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc list-inside text-sm space-y-2 text-slate-600 dark:text-slate-400">
                <li>User question analysis</li>
                <li>Data visualization generation</li>
                <li>Business report summarization</li>
                <li>Predictive model integration</li>
                <li>Trend analysis and KPI tracking</li>
              </ul>
            </CardContent>
          </Card>
        </div>

        <div className="max-w-4xl mx-auto">
          <Card>
            <CardHeader>
              <CardTitle>How It Works</CardTitle>
              <CardDescription>
                Our AI agent platform routes your questions to specialized agents for optimal responses
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col gap-6">
                <div className="flex items-start gap-4">
                  <div className="bg-slate-100 dark:bg-slate-800 rounded-full p-2 text-slate-500">1</div>
                  <div>
                    <h3 className="font-medium">Ask a question</h3>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      Type your question in natural language about code, documents, or business data
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="bg-slate-100 dark:bg-slate-800 rounded-full p-2 text-slate-500">2</div>
                  <div>
                    <h3 className="font-medium">Automatic routing</h3>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      Our system analyzes your question and routes it to the most appropriate agent
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="bg-slate-100 dark:bg-slate-800 rounded-full p-2 text-slate-500">3</div>
                  <div>
                    <h3 className="font-medium">Get specialized answers</h3>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      Receive tailored responses with code snippets, document summaries, or data visualizations
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
