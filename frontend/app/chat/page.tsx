"use client"

import React, { useState, useEffect, useRef, FormEvent, ChangeEvent, useCallback } from "react";
import { useRouter } from "next/navigation"
import { v4 as uuidv4 } from 'uuid';
import { Button } from "@/components/ui/button";
import { ChatSession, AgentType, ChatMessage as ApiChatMessage } from "../../lib/api/chat-service";
import { getUserChatSessions, createChatSessionInDB, getChatMessagesFromDB, saveChatMessageToDB, updateSessionAgentTypeInDB, deleteChatSessionFromDB, updateChatSessionTitleInDB } from "../../lib/api/postgres-chat-service";
import { createChatWebSocket, ChatWebSocketControls } from "../../lib/api/fastapi-service";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { SendIcon, Upload, FileIcon, Paperclip, XIcon } from "lucide-react"
import { ChatMessage as ChatMessageComponent } from "@/components/chat-message"
import { AgentSelector } from "@/components/agent-selector"
import { ChatHeader } from "@/components/chat-header"
import { Canvas } from "@/components/canvas"
import { ChatSidebar } from "@/components/chat-sidebar"
import { getCurrentUser, logoutUser, type User } from "@/lib/api/auth-service"
import { toast } from "@/hooks/use-toast"
import { isPreviewEnvironment } from "@/lib/api/mock-data"

import { Textarea } from "@/components/ui/textarea"

const AVAILABLE_AGENTS: (AgentType | "auto")[] = ["auto", "code", "analytics", "prediction", "supervisor"];

export default function ChatPage() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [ws, setWs] = useState<ChatWebSocketControls | null>(null);
  const [threadId, setThreadId] = useState<string>('');
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState<ApiChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isReceivingResponse, setIsReceivingResponse] = useState(false)
  const [selectedAgent, setSelectedAgent] = useState<AgentType | "auto">("auto")
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  const [isEditingTitle, setIsEditingTitle] = useState(false)
  const [editTitle, setEditTitle] = useState("")

  type CodeCanvasContent = { type: "code"; language: string; content: string; title: string }
  type DocumentCanvasContent = { type: "document"; data: any; title: string }
  type ChartCanvasContent = { type: "chart"; data: any; title: string }
  type CanvasContent = CodeCanvasContent | DocumentCanvasContent | ChartCanvasContent | null

  const [canvasContent, setCanvasContent] = useState<CanvasContent>(null)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSession, setActiveSession] = useState<ChatSession | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadTableName, setUploadTableName] = useState<string>("")
  const [createNewTable, setCreateNewTable] = useState(false)
  const [availableTables, setAvailableTables] = useState<string[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [attachedFile, setAttachedFile] = useState<File | null>(null)

  // Ref to keep track of the latest messages for callbacks
  const messagesRef = useRef<ApiChatMessage[]>(messages);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const readFileAsText = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = (error) => reject(error)
      reader.readAsText(file)
    })
  }

  // Initialize user data function, memoized with useCallback
  // This function is defined at the top-level of the ChatPage component.
  const initializeUserData = useCallback(async (user: User) => {
    if (user && user.id) {
      // Assuming fetchUserSessions is defined elsewhere (e.g., imported or a prop)
      // and is stable or correctly memoized if defined within ChatPage.
      await fetchUserSessions(user.id);
    }
  }, []); // If fetchUserSessions is defined in ChatPage and isn't stable/memoized, add it as a dependency here.

  useEffect(() => {
    const checkUser = async () => {
      const currentUser = await getCurrentUser();
      if (!currentUser) {
        if (isPreviewEnvironment()) {
          router.push("/demo");
          return null; // Ensure promise resolves with null if redirecting
        }
        router.push("/login");
        return null; // Ensure promise resolves with null if redirecting
      }
      setUser(currentUser);
      return currentUser;
    };

    checkUser().then((userData) => {
      if (userData) {
        initializeUserData(userData);
      }
    });
  }, [router, initializeUserData]); // Added initializeUserData to the dependency array

  useEffect(() => {
    if (activeSessionId) {
      const loadMessages = async () => {
        setIsLoading(true)
        const fetchedMessages = await getChatMessagesFromDB(activeSessionId)
        setMessages(fetchedMessages)
        setIsLoading(false)
      }
      loadMessages()
    } else {
      setMessages([])
    }
  }, [activeSessionId])

  useEffect(() => {
    if (messagesEndRef.current) {
      const container = messagesEndRef.current.parentElement
      if (container) {
        container.scrollTop = container.scrollHeight
      }
    }
  }, [messages])

  const fetchUserSessions = useCallback(async (userId: string): Promise<ChatSession[]> => {
    try {
      const sessions = await getUserChatSessions(userId)
      setSessions(sessions)

      const activeSession = sessions.find(s => !s.ended_at) || sessions[0] || null
      setActiveSession(activeSession)

      if (activeSession) {
        setEditTitle(activeSession.title || '')
        setActiveSessionId(activeSession.id)
        fetchSessionMessages(activeSession.id)
      }
      return sessions
    } catch (error) {
      console.error("Error fetching sessions:", error)
      toast({ title: "Error", description: "Failed to load chat sessions", variant: "destructive" })
      return []
    }
  }, []);

  const fetchSessionMessages = useCallback(async (sessionId: string) => {
    try {
      const messages = await getChatMessagesFromDB(sessionId)
      setMessages(messages)
    } catch (error) {
      console.error("Error fetching messages:", error)
      toast({ title: "Error", description: "Failed to load messages", variant: "destructive" })
    }
  }, []);

  const handleSessionChange = (sessionId: string) => {
    const selected = sessions.find(s => s.id === sessionId)
    if (selected) {
      setActiveSession(selected)
      setActiveSessionId(sessionId)
      setEditTitle(selected.title || '')
      fetchSessionMessages(sessionId)
      setCanvasContent(null)
    }
  }

  const handleEditTitle = () => {
    if (activeSession) {
      setEditTitle(activeSession.title || '')
      setIsEditingTitle(true)
    }
  }

  const handleSaveTitle = async () => {
    if (!activeSession || !editTitle.trim()) return
    try {
      const updatedSession = await updateChatSessionTitleInDB(activeSession.id, editTitle.trim())
      if (updatedSession) {
        setActiveSession(updatedSession)
        if (user && user.id) {
          await fetchUserSessions(user.id)
        }
      }
      setIsEditingTitle(false)
    } catch (error) {
      console.error("Error updating session title:", error)
      toast({ title: "Error", description: "Failed to update session title", variant: "destructive" })
    }
  }

  const handleCancelEditTitle = () => {
    setIsEditingTitle(false)
    if (activeSession) {
      setEditTitle(activeSession.title || '')
    }
  }

  const handleDeleteSession = async (sessionId: string) => {
    if (!confirm("Are you sure you want to delete this session?")) return
    try {
      const success = await deleteChatSessionFromDB(sessionId)
      if (success) {
        if (activeSession && activeSession.id === sessionId) {
          setActiveSession(null)
          setActiveSessionId(null)
          setMessages([])
        }
        if (user && user.id) {
          const newSessions = await fetchUserSessions(user.id)
          if (activeSession && activeSession.id === sessionId && newSessions.length > 0) {
            const newActiveSession = newSessions[0]
            setActiveSession(newActiveSession)
            setActiveSessionId(newActiveSession.id)
            fetchSessionMessages(newActiveSession.id)
            setEditTitle(newActiveSession.title || '')
          }
        }
        toast({ title: "Success", description: "Chat session deleted successfully." })
      }
    } catch (error) {
      console.error("Error deleting session:", error)
      toast({ title: "Error", description: "Failed to delete chat session", variant: "destructive" })
    }
  }

  const handleAgentChange = async (agentType: AgentType | "auto") => {
    setSelectedAgent(agentType)
    if (agentType !== "auto" && activeSessionId) {
      try {
        await updateSessionAgentTypeInDB(activeSessionId, agentType as AgentType)
        setActiveSession(prev => prev ? { ...prev, agent_type: agentType } : null);
        toast({ title: "Agent Updated", description: `Session agent set to ${agentType}` })
      } catch (error) {
        console.error("Error updating agent type:", error)
        toast({ title: "Error", description: "Failed to update agent type. Please try again.", variant: "destructive" })
      }
    }
  }

  const handleNewSession = useCallback(async (agentType: AgentType | "auto" = "auto") => {
    if (!user?.id) {
      toast({ title: "Error", description: "You must be logged in to start a new session.", variant: "destructive" });
      return null;
    }
    setIsLoading(true);
    try {
      const agentToStore = agentType === 'auto' ? 'code' : agentType;
      const newSession = await createChatSessionInDB(user.id, agentToStore);
      if (newSession) {
        await fetchUserSessions(user.id);
        handleSessionChange(newSession.id);
        toast({ title: "New Session", description: `New chat session started with ${agentType} agent.` });
        return newSession;
      }
    } catch (error) {
      console.error("Error creating new session:", error);
      toast({ title: "Error", description: "Failed to create new chat session", variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
    return null;
  }, [user, fetchUserSessions, handleSessionChange]);

  const handleSendMessage = useCallback(async (e: FormEvent) => {
    e.preventDefault();
    if (isLoading || (!input.trim() && !attachedFile)) return;

    setIsLoading(true);
    setIsReceivingResponse(true);

    let currentSessionId = activeSessionId;
    if (!currentSessionId) {
      const newSession = await handleNewSession(selectedAgent);
      if (!newSession) {
        setIsLoading(false);
        setIsReceivingResponse(false);
        return;
      }
      currentSessionId = newSession.id;
    }

    const userMessage: ApiChatMessage = {
      id: uuidv4(),
      session: currentSessionId,
      role: 'user',
      content: input,
      created_at: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMessage]);
    await saveChatMessageToDB(userMessage.session, userMessage.role, userMessage.content);

    const assistantMessageId = uuidv4();
    const assistantMessagePlaceholder: ApiChatMessage = {
      id: assistantMessageId,
      session: currentSessionId,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
    };
    setMessages(prev => [...prev, assistantMessagePlaceholder]);

    const messageToSend = input;
    const fileToSend = attachedFile;
    setInput("");
    setAttachedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";

    if (ws) ws.close();

    const newWsConnection = createChatWebSocket(
      currentSessionId,
      selectedAgent,
      (token) => { // onToken
        setMessages(prev => prev.map(msg =>
          msg.id === assistantMessageId ? { ...msg, content: msg.content + token } : msg
        ));
      },
      (agent) => { // onAgentChange
        toast({ title: "Agent Switched", description: `Request routed to ${agent} agent.` });
      },
      (tool) => { /* onToolStart */ console.log("Tool started:", tool); },
      (result) => { /* onToolEnd */ console.log("Tool ended:", result); },
      (error) => { // onError
        console.error("WebSocket Error:", error);
        setMessages(prev => prev.map(msg =>
          msg.id === assistantMessageId ? { ...msg, content: `${msg.content}\n\n**Error:** ${error}` } : msg
        ));
        setIsLoading(false);
        setIsReceivingResponse(false);
      },
      () => { // onDone
        const finalAssistantMessage = messagesRef.current.find(msg => msg.id === assistantMessageId);
        if (finalAssistantMessage) {
          saveChatMessageToDB(finalAssistantMessage.session, finalAssistantMessage.role, finalAssistantMessage.content);
        }
        setIsLoading(false);
        setIsReceivingResponse(false);
        if (newWsConnection) newWsConnection.close();
        setWs(null);
      }
    );

    setWs(newWsConnection);

    if (!newWsConnection) {
      toast({ title: "Connection Error", description: "Failed to create WebSocket connection.", variant: "destructive" });
      setIsLoading(false);
      setIsReceivingResponse(false);
      return;
    }

    let csvContent: string | null = null;
    if (fileToSend) {
      try {
        csvContent = await fileToSend.text();
      } catch (err) {
        console.error("Error reading file:", err);
        toast({ title: "File Read Error", description: "Could not read the attached file.", variant: "destructive" });
        setIsLoading(false);
        setIsReceivingResponse(false);
        newWsConnection.close();
        setWs(null);
        return;
      }
    }

    const payload = {
      user_query: messageToSend,
      csv_file_content: csvContent,
    };
    
    console.log(">>> CHAT_PAGE: Sending payload to WebSocket:", payload);
    newWsConnection.sendMessage(payload);

  }, [
    user,
    activeSessionId,
    selectedAgent,
    input,
    attachedFile,
    isLoading,
    ws,
    handleNewSession,
    messagesRef,
  ]);

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type !== "text/csv" && !file.name.endsWith('.csv')) {
        toast({ title: "Invalid File Type", description: "Only CSV files are allowed.", variant: "destructive" });
        return;
      }
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        toast({ title: "File Too Large", description: "Please select a file smaller than 10MB.", variant: "destructive" });
        return;
      }
      setAttachedFile(file);
    }
  };

  const handleRemoveAttachedFile = () => {
    setAttachedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleLogout = async () => {
    await logoutUser();
    router.push("/login");
  };

  const handleOpenUploadModal = async () => {
    toast({ title: "Feature Disabled", description: "Direct database upload is temporarily disabled." });
    // setIsUploadModalOpen(true); // Logic can be re-enabled later
  };

  const handleFileUpload = async () => {
    if (!uploadFile || !uploadTableName) return;
    setIsUploading(true);
    try {
      toast({ title: "Feature Disabled", description: "File upload is temporarily disabled." });
      setIsUploadModalOpen(false);
      setUploadFile(null);
      setUploadTableName("");
    } catch (error: any) {
      console.error("Error uploading file:", error);
      toast({ title: "Upload Failed", description: error.message || "An unknown error occurred.", variant: "destructive" });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-900">
      <ChatSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        user={user}
        onLogout={handleLogout}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSessionChange={handleSessionChange}
        onNewSession={handleNewSession}
        isLoading={isLoading}
      />
      <div className="flex flex-1 flex-col">
        {user && (
          <ChatHeader
            user={{ ...user, isDemo: isPreviewEnvironment() }}
            onLogout={handleLogout}
            onMenuClick={() => setIsSidebarOpen(!isSidebarOpen)}
            activeSession={activeSession}
            isEditingTitle={isEditingTitle}
            editTitle={editTitle}
            onEditTitle={handleEditTitle}
            onSaveTitle={handleSaveTitle}
            onCancelEditTitle={handleCancelEditTitle}
            onDeleteSession={handleDeleteSession}
            onEditTitleChange={(e) => setEditTitle(e.target.value)}
          />
        )}
        <main className="flex-1 overflow-auto p-4">
          <div className="h-full space-y-4">
            {messages.map((m, i) => (
              <ChatMessageComponent key={m.id || i} message={{ id: m.id, role: m.role, content: m.content, agentType: m.agentType }} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </main>
        <footer className="border-t p-4 bg-white dark:bg-slate-900">
          <form onSubmit={handleSendMessage} className="relative">
            {attachedFile && (
              <div className="absolute bottom-full left-0 mb-2 flex items-center gap-2 rounded-md bg-slate-100 p-2 dark:bg-slate-800">
                <FileIcon className="h-5 w-5 text-slate-500" />
                <span className="text-sm text-slate-700 dark:text-slate-300">{attachedFile.name}</span>
                <Button variant="ghost" size="icon" onClick={handleRemoveAttachedFile} className="h-6 w-6">
                  <XIcon className="h-4 w-4" />
                </Button>
              </div>
            )}
            <Textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message here..."
              className="w-full rounded-lg border border-slate-300 p-2 pr-24 dark:border-slate-700 dark:bg-slate-800"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(e as any);
                }
              }}
              rows={1}
              disabled={isLoading}
            />
            <div className="absolute bottom-2 right-2 flex items-center gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                >
                  <Paperclip className="h-5 w-5" />
                </Button>
                <input type="file" ref={fileInputRef} onChange={handleFileSelect} className="hidden" accept=".csv" />
                <Button type="submit" disabled={isLoading || !activeSessionId || (!input.trim() && !attachedFile)}>
                  <SendIcon className="h-5 w-5" />
                </Button>
              </div>
          </form>
        </footer>
      </div>
      {canvasContent && <Canvas content={canvasContent} onClose={() => setCanvasContent(null)} />}
      <Dialog open={isUploadModalOpen} onOpenChange={setIsUploadModalOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Upload CSV to Database</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <label htmlFor="file-upload" className="mb-2 block text-sm font-medium">
                CSV File
              </label>
              <Input
                id="file-upload"
                type="file"
                accept=".csv"
                onChange={(e) => setUploadFile(e.target.files ? e.target.files[0] : null)}
              />
            </div>
            <div>
              <label htmlFor="tableNameLabel" className="mb-2 block text-sm font-medium">
                Table Name
              </label>
              <div className="mt-2 flex items-center mb-2">
                <input
                  type="checkbox"
                  id="createNewTable"
                  checked={createNewTable}
                  onChange={(e) => {
                    setCreateNewTable(e.target.checked);
                    if (e.target.checked) {
                        setUploadTableName("");
                    } else if (availableTables.length > 0) {
                        setUploadTableName(availableTables[0] || "");
                    } else {
                        setUploadTableName("");
                    }
                  }}
                  className="mr-2"
                />
                <label htmlFor="createNewTable" className="text-sm">
                  Create new table
                </label>
              </div>

              {createNewTable ? (
                <Input
                  id="tableNameInput"
                  value={uploadTableName}
                  onChange={(e) => setUploadTableName(e.target.value)}
                  placeholder="Enter new table name (e.g., sales_data_q1)"
                />
              ) : (
                <select
                  id="tableSelect"
                  value={uploadTableName}
                  onChange={(e) => setUploadTableName(e.target.value)}
                  className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                  disabled={availableTables.length === 0}
                >
                  {availableTables.length === 0 ? (
                    <option value="" disabled>No existing tables found</option>
                  ) : (
                    availableTables.map((table) => (
                      <option key={table} value={table}>
                        {table}
                      </option>
                    ))
                  )}
                </select>
              )}
            </div>
            <div className="flex justify-end space-x-2 pt-4">
              <Button variant="outline" onClick={() => setIsUploadModalOpen(false)}>Cancel</Button>
              <Button onClick={handleFileUpload} disabled={isUploading || !uploadTableName || !uploadFile}>
                {isUploading ? 'Uploading...' : 'Upload'}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
