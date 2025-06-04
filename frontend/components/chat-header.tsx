"use client"

import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { MenuIcon, LogOutIcon, UserIcon } from "lucide-react"
import Link from "next/link"

export function ChatHeader({ user, onMenuClick, onLogout }) {
  return (
    <header className="border-b h-14 flex items-center justify-between px-4 bg-white dark:bg-slate-900">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" onClick={onMenuClick} className="md:hidden">
          <MenuIcon className="h-5 w-5" />
          <span className="sr-only">Menu</span>
        </Button>
        <h1 className="text-lg font-semibold">AI Agent Platform</h1>
        {user.isDemo && <div className="text-xs bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full">Demo Mode</div>}
      </div>

      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" asChild className="hidden md:flex">
          <Link href="/profile">
            <UserIcon className="h-5 w-5" />
            <span className="sr-only">Profile</span>
          </Link>
        </Button>

        <Button variant="ghost" size="icon" onClick={onLogout} className="hidden md:flex">
          <LogOutIcon className="h-5 w-5" />
          <span className="sr-only">Log out</span>
        </Button>

        <div className="flex items-center gap-2">
          <span className="text-sm hidden md:inline">{user.name}</span>
          <Link href="/profile">
            <Avatar className="h-8 w-8 cursor-pointer">
              <AvatarImage src={user.avatar || "/placeholder.svg"} />
              <AvatarFallback>{user.name.charAt(0)}</AvatarFallback>
            </Avatar>
          </Link>
        </div>
      </div>
    </header>
  )
}
