'use client';

import { useState, useEffect, useRef, FormEvent } from 'react';
import { marked } from 'marked';
import Chart from 'chart.js/auto';

// --- 유틸리티 함수 ---
function getCookie(name: string): string | null {
    if (typeof document === 'undefined') return null;
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) {
        return parts.pop()?.split(';').shift() || null;
    }
    return null;
}

// --- 타입 정의 ---
type TMessage = {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    createdAt: string;
    tool_calls?: TToolCall[];
};

type TToolCall = {
    name: string;
    args: any;
    output: any;
};

type TSession = {
    id: string;
    title: string;
};

// --- 컴포넌트 ---
export default function ChatPage() {
    const [sessions, setSessions] = useState<TSession[]>([]);
    const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
    const [messages, setMessages] = useState<TMessage[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isSidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [isChartModalOpen, setChartModalOpen] = useState(false);
    const [chartContent, setChartContent] = useState<{ canvas_html: string; script_js: string } | null>(null);
    
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const chartModalContentRef = useRef<HTMLDivElement>(null);

    // --- 데이터 페칭 ---
    useEffect(() => {
        // Django API로부터 세션 목록과 초기 메시지를 가져오는 로직
        const fetchSessions = async () => {
            try {
                const response = await fetch('/conversations/api/v1/sessions/');
                if (!response.ok) {
                    throw new Error('Failed to fetch sessions');
                }
                const data: TSession[] = await response.json();
                setSessions(data);
                if (data.length > 0) {
                    setActiveSessionId(data[0].id);
                }
            } catch (error) {
                console.error('Error fetching sessions:', error);
                // Optionally, set an error state to show in the UI
            }
        };

        fetchSessions();
    }, []);

    useEffect(() => {
        if (activeSessionId) {
            // 특정 세션의 메시지를 가져오는 로직
            const fetchMessages = async () => {
                try {
                    const response = await fetch(`/conversations/api/v1/sessions/${activeSessionId}/messages/`);
                    if (!response.ok) {
                        throw new Error('Failed to fetch messages');
                    }
                    const data: TMessage[] = await response.json();
                    setMessages(data);
                } catch (error) {
                    console.error('Error fetching messages:', error);
                    // Optionally, set an error state to show in the UI
                }
            };

            fetchMessages();
        }
    }, [activeSessionId]);

    // --- UI 효과 ---
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    useEffect(() => {
        const renderMarkdown = async () => {
            for (const elem of Array.from(document.querySelectorAll('.markdown-content'))) {
                const htmlElement = elem as HTMLElement;
                htmlElement.innerHTML = await marked.parse(htmlElement.innerText);
            }
        };
        renderMarkdown().catch(console.error);
    }, [messages]);

    useEffect(() => {
        if (isChartModalOpen && chartContent && chartModalContentRef.current) {
            chartModalContentRef.current.innerHTML = chartContent.canvas_html;
            const script = document.createElement('script');
            script.textContent = `(function() { ${chartContent.script_js} })();`;
            chartModalContentRef.current.appendChild(script);
        }
    }, [isChartModalOpen, chartContent]);


    // --- 이벤트 핸들러 ---

    const handleNewSession = () => {
        const newSession: TSession = { id: `session-${Date.now()}`, title: '새로운 채팅' };
        setSessions([newSession, ...sessions]);
        setActiveSessionId(newSession.id);
        setMessages([]);
    };

    const handleDeleteSession = (sessionId: string) => {
        setSessions(sessions.filter(s => s.id !== sessionId));
        if (activeSessionId === sessionId) {
            setActiveSessionId(sessions.length > 1 ? sessions[1].id : null);
            setMessages([]);
        }
    };

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (!input.trim() || !activeSessionId) return;

        const userMessage: TMessage = {
            id: `user-${Date.now()}`,
            role: 'user',
            content: input,
            createdAt: new Date().toISOString(),
        };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        if (!activeSessionId) {
            console.error("No active session selected. Cannot send message.");
            // Restore input and remove optimistic message
            setInput(userMessage.content);
            setMessages(prev => prev.filter(m => m.id !== userMessage.id));
            setIsLoading(false);
            return;
        }

        const response = await fetch(`/conversations/api/v1/sessions/${activeSessionId}/invoke/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken') || '',
            },
            body: JSON.stringify({ 'input': input }),
        });

        if (!response.body) return;
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantMessageId = `assistant-${Date.now()}`;
        let currentAiMessage: TMessage = { id: assistantMessageId, role: 'assistant', content: '', createdAt: new Date().toISOString(), tool_calls: [] };
        setMessages(prev => [...prev, currentAiMessage]);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n\n');

            for (const line of lines) {
                if (line.startsWith('data:')) {
                    try {
                        const json = JSON.parse(line.substring(5));
                        if (json.event === 'message_chunk') {
                            setMessages(prev => prev.map(msg => 
                                msg.id === assistantMessageId ? { ...msg, content: msg.content + json.data } : msg
                            ));
                        } else if (json.event === 'tool_update') {
                            // Tool UI 업데이트 로직
                            setMessages(prev => prev.map(msg => 
                                msg.id === assistantMessageId ? { ...msg, tool_calls: json.data.assistant.messages } : msg
                            ));
                            // 차트 버튼 주입 로직
                            const toolMessage = json.data.assistant.messages.find((m: any) => m.type === 'tool' && (m.name === 'ChartGenerator' || m.name === 'analyst_chart_tool'));
                            if (toolMessage) {
                                const chartData = JSON.parse(toolMessage.content);
                                setChartContent(chartData);
                            }
                        } else if (json.event === 'stream_end') {
                            setIsLoading(false);
                        }
                    } catch (error) {
                        console.error('Error parsing stream data:', error, 'line:', line);
                    }
                }
            }
        }
    };

    // --- 렌더링 ---
    return (
        <div className="flex h-screen overflow-hidden bg-white text-gray-800">
            {/* 세션 사이드바 */}
            <aside className={`bg-white border-r border-gray-200 h-full flex flex-col transition-all duration-300 ${isSidebarCollapsed ? 'w-20' : 'w-80'}`}>
                <div className="p-4 border-b border-gray-200 flex justify-between items-center">
                    <h2 className={`text-xl font-bold text-gray-800 ${isSidebarCollapsed ? 'hidden' : ''}`}>채팅 세션</h2>
                    <button onClick={() => setSidebarCollapsed(!isSidebarCollapsed)} className="text-gray-500 hover:text-blue-600">
                        <i className={`fas fa-chevron-left transform transition-transform duration-300 ${isSidebarCollapsed ? 'rotate-180' : ''}`}></i>
                    </button>
                </div>
                <div className="p-4 border-b border-gray-200">
                    <button onClick={handleNewSession} className="flex items-center justify-center w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition-colors">
                        <i className="fas fa-plus-circle"></i>
                        <span className={`ml-2 ${isSidebarCollapsed ? 'hidden' : ''}`}>새 세션 시작</span>
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-2">
                    {sessions.map(session => (
                        <a key={session.id} href="#" onClick={() => setActiveSessionId(session.id)} 
                           className={`flex items-center p-3 rounded-lg transition-all duration-200 ${activeSessionId === session.id ? 'bg-blue-100 border-l-4 border-blue-600 text-blue-600' : 'hover:bg-gray-50 text-gray-600 border-l-4 border-transparent'}`}>
                            <i className={`fas fa-comment-dots fa-fw ${activeSessionId === session.id ? 'text-blue-600' : 'text-gray-400'}`}></i>
                            <span className={`flex-1 truncate ml-3 font-medium ${isSidebarCollapsed ? 'hidden' : ''}`}>{session.title}</span>
                        </a>
                    ))}
                </div>
            </aside>

            {/* 채팅창 */}
            <main className="flex-1 flex flex-col bg-white">
                <header className="bg-white/95 backdrop-blur-sm border-b border-gray-200 p-4 flex justify-between items-center sticky top-0 z-10">
                    <div className="flex items-center">
                        <h2 className="text-lg font-semibold truncate text-gray-800">
                            {sessions.find(s => s.id === activeSessionId)?.title || '새로운 채팅'}
                        </h2>
                        {activeSessionId && (
                            <button onClick={() => handleDeleteSession(activeSessionId)} className="ml-3 p-2 text-gray-500 rounded-md hover:bg-gray-100 hover:text-red-500">
                                <i className="fas fa-trash-alt"></i>
                            </button>
                        )}
                    </div>
                    <div className="flex items-center space-x-4">
                        <span className="text-sm text-gray-600">안녕하세요, 사용자님</span>
                        <a href="#" className="text-sm bg-red-600 hover:bg-red-700 text-white px-3 py-1.5 rounded-lg transition-colors">로그아웃</a>
                    </div>
                </header>

                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {messages.map(msg => (
                        <div key={msg.id}>
                            {msg.role === 'user' ? (
                                <div className="flex justify-end">
                                    <div className="max-w-xl">
                                        <div className="chat-bubble-user text-white p-4 rounded-2xl rounded-tr-none shadow-lg">
                                            <p>{msg.content}</p>
                                        </div>
                                        <p className="text-xs text-gray-500 text-right mt-1">{new Date(msg.createdAt).toLocaleTimeString()}</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex items-start gap-3 mb-4">
                                    <div className="w-full max-w-xl">
                                        {msg.tool_calls && msg.tool_calls.length > 0 && (
                                            <div className="tool-usage-container">
                                                <details open>
                                                    <summary><i className="fas fa-cogs fa-fw"></i>Tool Usage</summary>
                                                    <div className="tool-usage-content">
                                                        {msg.tool_calls.map((call, index) => (
                                                            <div key={index}>
                                                                <div className="tool-call-item">
                                                                    <div className="font-semibold"><i className="fas fa-wrench fa-fw"></i> {call.name}</div>
                                                                    <pre>{JSON.stringify(call.args, null, 2)}</pre>
                                                                </div>
                                                                <div className="tool-output-item">
                                                                    <div className="font-semibold"><i className="fas fa-clipboard-check fa-fw"></i> Output</div>
                                                                    <pre>{JSON.stringify(call.output, null, 2)}</pre>
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </details>
                                            </div>
                                        )}
                                        <div className="chat-bubble-ai text-gray-800 p-4 rounded-2xl rounded-tl-none shadow-lg bg-gray-50">
                                            <div className="markdown-content" dangerouslySetInnerHTML={{ __html: marked.parse(msg.content) }}></div>
                                        </div>
                                        <p className="text-xs text-gray-500 mt-1">{new Date(msg.createdAt).toLocaleTimeString()}</p>
                                        {chartContent && (
                                            <div className="mt-2">
                                                <button onClick={() => setChartModalOpen(true)} className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700 transition-colors flex items-center">
                                                    <i className="fas fa-chart-pie fa-fw mr-2"></i> Open Canvas
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex items-center">
                             <div className="typing-indicator">
                                <span></span><span></span><span></span>
                            </div>
                            <span className="ml-2 text-gray-400">생성 중...</span>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <div className="bg-white/95 backdrop-blur-sm border-t border-gray-200 p-4 sticky bottom-0">
                    <form onSubmit={handleSubmit} className="flex items-center gap-2 max-w-4xl mx-auto">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="업무 내용을 입력하세요..."
                            className="flex-1 bg-white border border-gray-200 rounded-full py-3 px-5 focus:outline-none focus:ring-2 focus:ring-blue-600 disabled:opacity-50 text-gray-800 shadow-sm"
                            disabled={!activeSessionId || isLoading}
                        />
                        <button type="submit" className="bg-gradient-to-r from-blue-600 to-blue-500 w-14 h-14 rounded-full flex items-center justify-center flex-shrink-0 transition-all disabled:opacity-50"
                            disabled={!activeSessionId || isLoading}>
                            <i className="fas fa-paper-plane text-white text-xl"></i>
                        </button>
                    </form>
                </div>
            </main>

            {/* Chart Modal */}
            <div className={`fixed top-0 right-0 h-full w-full md:w-1/2 lg:w-1/3 bg-white shadow-2xl z-50 p-6 border-l border-gray-200 flex flex-col transform transition-transform duration-300 ease-in-out ${isChartModalOpen ? 'translate-x-0' : 'translate-x-full'}`}>
                 <div className="flex justify-between items-center pb-4 border-b border-gray-200">
                    <h3 className="text-xl font-bold text-gray-800">Chart View</h3>
                    <button onClick={() => setChartModalOpen(false)} className="text-gray-500 hover:text-gray-800">
                        <i className="fas fa-times fa-lg"></i>
                    </button>
                </div>
                <div ref={chartModalContentRef} className="flex-1 overflow-y-auto mt-4"></div>
            </div>
        </div>
    );
}
