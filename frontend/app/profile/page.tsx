"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Switch } from "@/components/ui/switch"
import { GithubIcon, ArrowLeftIcon, CheckIcon } from "lucide-react"
import Link from "next/link"
import { getCurrentUser, updateUserProfile, type User } from "@/lib/api/auth-service"
import { addRepository } from "@/lib/api/code-service"

export default function ProfilePage() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [isGithubConnected, setIsGithubConnected] = useState(false)
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [isSaved, setIsSaved] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [repoUrl, setRepoUrl] = useState("")

  useEffect(() => {
    // Check if user is logged in
    const checkUser = async () => {
      const currentUser = await getCurrentUser()
      if (!currentUser) {
        router.push("/login")
        return
      }

      setUser(currentUser)
      setName(currentUser.name || "")
      setEmail(currentUser.email || "")
    }

    checkUser()
  }, [router])

  const handleSaveProfile = async () => {
    if (!user) return

    setIsLoading(true)

    // Update user profile using the auth service
    // This connects to the User model in the Accounts section
    const updatedUser = await updateUserProfile({ name, email })

    if (updatedUser) {
      setUser(updatedUser)
      // Show saved indicator
      setIsSaved(true)
      setTimeout(() => setIsSaved(false), 2000)
    }

    setIsLoading(false)
  }

  const connectGithub = () => {
    // In a real implementation, this would redirect to GitHub OAuth
    // For demo purposes, we'll just set the state
    setIsGithubConnected(true)
  }

  const disconnectGithub = () => {
    // In a real implementation, this would revoke GitHub access
    // For demo purposes, we'll just set the state
    setIsGithubConnected(false)
  }

  const handleAddRepository = async () => {
    if (!repoUrl) return

    setIsLoading(true)

    // Add repository using the code service
    // This connects to the GitRepository model in the Knowledge section
    const repo = await addRepository(repoUrl)

    if (repo) {
      setRepoUrl("")
      // Show success message
      setIsSaved(true)
      setTimeout(() => setIsSaved(false), 2000)
    }

    setIsLoading(false)
  }

  if (!user) {
    return <div className="min-h-screen flex items-center justify-center">Loading...</div>
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <Button variant="ghost" asChild className="gap-2">
            <Link href="/chat">
              <ArrowLeftIcon className="h-4 w-4" />
              Back to Chat
            </Link>
          </Button>
        </div>

        <div className="max-w-2xl mx-auto">
          <h1 className="text-2xl font-bold mb-6">My Profile</h1>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Profile Information</CardTitle>
                <CardDescription>Update your personal information</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-4 mb-6">
                  <Avatar className="h-16 w-16">
                    <AvatarImage src={user.avatar || "/placeholder.svg?height=64&width=64"} />
                    <AvatarFallback>{name.charAt(0)}</AvatarFallback>
                  </Avatar>
                  <div>
                    <Button variant="outline" size="sm">
                      Change Avatar
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="name">Name</Label>
                  <Input id="name" value={name} onChange={(e) => setName(e.target.value)} />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} disabled />
                  <p className="text-xs text-slate-500">
                    Email cannot be changed. Contact your administrator for assistance.
                  </p>
                </div>
              </CardContent>
              <CardFooter>
                <Button onClick={handleSaveProfile} className="gap-2" disabled={isLoading}>
                  {isSaved && <CheckIcon className="h-4 w-4" />}
                  {isSaved ? "Saved" : "Save Changes"}
                </Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>GitHub Integration</CardTitle>
                <CardDescription>Connect your GitHub account to enable code repository access</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {isGithubConnected ? (
                  <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-900 rounded-md p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <GithubIcon className="h-5 w-5" />
                      <div>
                        <div className="font-medium">GitHub Connected</div>
                        <div className="text-sm text-slate-500 dark:text-slate-400">Connected as {name}</div>
                      </div>
                    </div>
                    <Button variant="outline" size="sm" onClick={disconnectGithub}>
                      Disconnect
                    </Button>
                  </div>
                ) : (
                  <div className="bg-slate-50 dark:bg-slate-800 border rounded-md p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <GithubIcon className="h-5 w-5" />
                      <div>
                        <div className="font-medium">Connect to GitHub</div>
                        <div className="text-sm text-slate-500 dark:text-slate-400">
                          Enable repository access for code analysis
                        </div>
                      </div>
                    </div>
                    <Button onClick={connectGithub}>Connect</Button>
                  </div>
                )}

                {isGithubConnected && (
                  <div className="mt-4">
                    <Label htmlFor="repo-url">Add Repository</Label>
                    <div className="flex gap-2 mt-2">
                      <Input
                        id="repo-url"
                        placeholder="https://github.com/username/repo"
                        value={repoUrl}
                        onChange={(e) => setRepoUrl(e.target.value)}
                      />
                      <Button onClick={handleAddRepository} disabled={isLoading || !repoUrl}>
                        Add
                      </Button>
                    </div>
                    <p className="text-xs text-slate-500 mt-1">
                      This will index the repository for code search and analysis
                    </p>
                  </div>
                )}

                <div className="space-y-4 mt-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">Repository Access</div>
                      <div className="text-sm text-slate-500 dark:text-slate-400">
                        Allow access to private repositories
                      </div>
                    </div>
                    <Switch disabled={!isGithubConnected} />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium">Code Analysis</div>
                      <div className="text-sm text-slate-500 dark:text-slate-400">
                        Enable AI code analysis on your repositories
                      </div>
                    </div>
                    <Switch disabled={!isGithubConnected} />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
