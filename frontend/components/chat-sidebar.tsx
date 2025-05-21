"use client"

import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Separator } from "@/components/ui/separator"
import { CodeIcon, FileTextIcon, BarChart3Icon, HomeIcon, LogOutIcon, XIcon } from "lucide-react"
import Link from "next/link"

export function ChatSidebar({ isOpen, onClose, user, onLogout }) {
  return (
    <>
      {/* Desktop Sidebar */}
      <div className="w-64 border-r bg-white dark:bg-slate-900 hidden md:flex flex-col">
        <div className="p-4 border-b">
          <h2 className="font-semibold">AI Agent Platform</h2>
        </div>

        <div className="flex-1 overflow-auto py-2">
          <div className="px-3 py-2">
            <h3 className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">NAVIGATION</h3>
            <div className="space-y-1">
              <Link
                href="/"
                className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800"
              >
                <HomeIcon className="h-4 w-4" />
                Home
              </Link>
            </div>
          </div>

          <Separator className="my-2" />

          <div className="px-3 py-2">
            <h3 className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">AGENTS</h3>
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer">
                <CodeIcon className="h-4 w-4 text-blue-500" />
                Code Analysis
              </div>
              <div className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer">
                <FileTextIcon className="h-4 w-4 text-purple-500" />
                Document QA
              </div>
              <div className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer">
                <BarChart3Icon className="h-4 w-4 text-green-500" />
                Business Analysis
              </div>
            </div>
          </div>
        </div>

        <div className="border-t p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Avatar className="h-8 w-8">
                <AvatarImage src={user?.avatar || "/placeholder.svg"} />
                <AvatarFallback>{user?.name?.charAt(0)}</AvatarFallback>
              </Avatar>
              <div>
                <div className="text-sm font-medium">{user?.name}</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">{user?.email}</div>
              </div>
            </div>
            <Button variant="ghost" size="icon" onClick={onLogout}>
              <LogOutIcon className="h-4 w-4" />
              <span className="sr-only">Log out</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Sidebar */}
      <Sheet open={isOpen} onOpenChange={onClose}>
        <SheetContent side="left" className="w-[300px] sm:w-[350px] p-0">
          <SheetHeader className="p-4 border-b">
            <SheetTitle className="flex items-center justify-between">
              <span>AI Agent Platform</span>
              <Button variant="ghost" size="icon" onClick={onClose}>
                <XIcon className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </Button>
            </SheetTitle>
          </SheetHeader>

          <div className="overflow-auto py-2">
            <div className="px-3 py-2">
              <h3 className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">NAVIGATION</h3>
              <div className="space-y-1">
                <Link
                  href="/"
                  className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  <HomeIcon className="h-4 w-4" />
                  Home
                </Link>
              </div>
            </div>

            <Separator className="my-2" />

            <div className="px-3 py-2">
              <h3 className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">AGENTS</h3>
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer">
                  <CodeIcon className="h-4 w-4 text-blue-500" />
                  Code Analysis
                </div>
                <div className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer">
                  <FileTextIcon className="h-4 w-4 text-purple-500" />
                  Document QA
                </div>
                <div className="flex items-center gap-2 text-sm rounded-md px-3 py-2 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer">
                  <BarChart3Icon className="h-4 w-4 text-green-500" />
                  Business Analysis
                </div>
              </div>
            </div>
          </div>

          <div className="border-t p-4 mt-auto">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Avatar className="h-8 w-8">
                  <AvatarImage src={user?.avatar || "/placeholder.svg"} />
                  <AvatarFallback>{user?.name?.charAt(0)}</AvatarFallback>
                </Avatar>
                <div>
                  <div className="text-sm font-medium">{user?.name}</div>
                  <div className="text-xs text-slate-500 dark:text-slate-400">{user?.email}</div>
                </div>
              </div>
              <Button variant="ghost" size="icon" onClick={onLogout}>
                <LogOutIcon className="h-4 w-4" />
                <span className="sr-only">Log out</span>
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
