"use client"

import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Input } from "@/components/ui/input"
import { MenuIcon, LogOutIcon, UserIcon, EditIcon, TrashIcon, SaveIcon, XIcon } from "lucide-react"
import Link from "next/link"

interface ChatHeaderProps {
  user: {
    name: string;
    avatar?: string;
    isDemo?: boolean;
  };
  onMenuClick: () => void;
  onLogout: () => void;
  activeSession?: {
    id: string;
    title: string;
  } | null;
  isEditingTitle?: boolean;
  editTitle?: string;
  onEditTitle?: () => void;
  onSaveTitle?: () => void;
  onCancelEditTitle?: () => void;
  onDeleteSession?: (sessionId: string) => void;
  onEditTitleChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export function ChatHeader({ 
  user, 
  onMenuClick, 
  onLogout,
  activeSession,
  isEditingTitle,
  editTitle,
  onEditTitle,
  onSaveTitle,
  onCancelEditTitle,
  onDeleteSession,
  onEditTitleChange
}: ChatHeaderProps) {
  return (
    <header className="border-b h-14 flex items-center justify-between px-4 bg-white dark:bg-slate-900">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" onClick={onMenuClick} className="md:hidden">
          <MenuIcon className="h-5 w-5" />
          <span className="sr-only">Menu</span>
        </Button>
        
        {/* App Title or Session Title with Edit Mode */}
        {!activeSession ? (
          <h1 className="text-lg font-semibold">AI Agent Platform</h1>
        ) : isEditingTitle ? (
          <div className="flex items-center gap-2">
            <Input 
              value={editTitle} 
              onChange={onEditTitleChange}
              className="h-8 w-52 max-w-xs" 
              placeholder="세션 제목 입력..."
              autoFocus
            />
            <Button variant="ghost" size="icon" onClick={onSaveTitle} title="저장">
              <SaveIcon className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={onCancelEditTitle} title="취소">
              <XIcon className="h-4 w-4" />
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">{activeSession.title || "새 채팅 세션"}</h1>
            <Button variant="ghost" size="icon" onClick={onEditTitle} title="제목 편집">
              <EditIcon className="h-4 w-4" />
            </Button>
            <Button 
              variant="ghost" 
              size="icon" 
              onClick={() => onDeleteSession && onDeleteSession(activeSession.id)}
              title="세션 삭제"
            >
              <TrashIcon className="h-4 w-4 text-red-500" />
            </Button>
          </div>
        )}

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
