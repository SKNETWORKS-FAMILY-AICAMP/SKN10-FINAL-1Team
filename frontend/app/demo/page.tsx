"use client"

import { useRouter } from "next/navigation"
import { useEffect } from "react"

export default function DemoPage() {
  const router = useRouter()

  useEffect(() => {
    // Set a demo user for the demo experience
    localStorage.setItem(
      "user",
      JSON.stringify({
        id: "demo123",
        name: "Demo User",
        email: "demo@example.com",
        avatar: "/placeholder.svg?height=40&width=40",
        isDemo: true,
      }),
    )

    // Redirect to chat interface
    router.push("/chat")
  }, [router])

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="animate-pulse">Loading demo...</div>
    </div>
  )
}
