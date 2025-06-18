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
import { addRepository, connectGithubWithToken } from "@/lib/api/code-service" // connectGithubWithToken 추가
import GithubAuthModal from "@/components/github-auth-modal"; // GithubAuthModal 추가

export default function ProfilePage() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [isGithubConnected, setIsGithubConnected] = useState(false)
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [isSaved, setIsSaved] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [repoUrl, setRepoUrl] = useState("")
  const [showGithubAuthModal, setShowGithubAuthModal] = useState(false); // 모달 상태 추가

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
    // setIsGithubConnected(true) // 기존 로직 주석 처리
    setShowGithubAuthModal(true); // 모달을 열도록 변경
  }

  const disconnectGithub = () => {
    // In a real implementation, this would revoke GitHub access
    // For demo purposes, we'll just set the state
    setIsGithubConnected(false)
    // TODO: 백엔드에 GitHub 연동 해제 API 호출 추가 필요
  }

  const handleTokenSubmit = async (token: string) => {
    if (!user) return;
    setIsLoading(true);
    try {
      // 백엔드 API를 호출하여 토큰을 저장하고 연동 상태를 업데이트합니다.
      // 예시: await connectGithubWithToken(user.id, token);
      // 아래는 임시로 상태만 변경합니다.
      console.log('Submitting token:', token); // 실제 API 호출로 대체 필요
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
      setIsGithubConnected(true);
      setShowGithubAuthModal(false);
      // 성공 알림 등을 추가할 수 있습니다.
    } catch (error) {
      console.error("Failed to connect GitHub with token:", error);
      // 사용자에게 에러 메시지를 표시합니다.
      throw error; // 모달에서 에러를 처리할 수 있도록 다시 throw
    }
    setIsLoading(false);
  };

  const handleOAuthStart = () => {
    // TODO: 백엔드의 GitHub OAuth 시작 API를 호출하여 GitHub 인증 페이지로 리다이렉션합니다.
    console.log('Starting GitHub OAuth flow...');
    // 예시: window.location.href = '/api/github/oauth/login/';
    // 현재는 모달만 닫습니다.
    setShowGithubAuthModal(false);
    alert('GitHub OAuth 기능은 현재 개발 중입니다.');
  };

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

                {/* GitHub 인증 모달 추가 */}
                <GithubAuthModal
                  isOpen={showGithubAuthModal}
                  onClose={() => setShowGithubAuthModal(false)}
                  onTokenSubmit={handleTokenSubmit}
                  onOAuthStart={handleOAuthStart}
                />

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
