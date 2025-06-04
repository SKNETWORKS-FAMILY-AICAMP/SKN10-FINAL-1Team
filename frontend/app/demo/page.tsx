"use client"

import { useRouter } from "next/navigation"
import { useEffect } from "react"
import { mockUser } from "@/lib/api/mock-data"

export default function DemoPage() {
  const router = useRouter()

  useEffect(() => {
    // Set a demo user for the demo experience
    localStorage.setItem(
      "user",
      JSON.stringify({
        ...mockUser,
        isDemo: true,
      }),
    )

    // Set a mock token
    localStorage.setItem("token", "demo-token-123")

    // Redirect to chat interface
    router.push("/chat")
  }, [router])

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="animate-pulse">Loading demo...</div>
    </div>
  )
}
