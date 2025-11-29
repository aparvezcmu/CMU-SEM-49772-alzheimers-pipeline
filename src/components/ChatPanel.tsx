import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Wifi, WifiOff, Upload, FileText, X, Loader } from 'lucide-react';
import { ChatMessage } from '../types';
import { ChatWebSocket, ChatMessage as WSChatMessage } from '../lib/chat-websocket';
import { MarkdownMessage } from './MarkdownMessage';

interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (content: string, role?: 'user' | 'assistant') => void;
  currentCode?: string; // Current cell code (legacy, may be removed)
  currentCellId?: string; // Current active cell ID
  allEditedCodes?: Record<string, string>; // All edited codes from UI
  initialCodes?: Record<string, string>; // Initial codes from file
}

interface PdfInfo {
  name: string;
  numPages: number;
  numChunks: number;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({ messages, onSendMessage, currentCode, currentCellId, allEditedCodes, initialCodes }) => {
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [currentMessage, setCurrentMessage] = useState('');
  const [uploadedPdf, setUploadedPdf] = useState<PdfInfo | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [sessionId] = useState(() => {
    // Generate session ID once on mount
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<ChatWebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const handleErrorMessage = (event: any) => {
      const { message } = event.detail;
      if (message) {
        sendMessageToAI(message, false); // false means don't add to messages (already added)
      }
    };

    window.addEventListener('sendErrorMessage', handleErrorMessage);
    
    return () => {
      window.removeEventListener('sendErrorMessage', handleErrorMessage);
    };
  }, [messages]); // Include messages in dependency to get latest history

  useEffect(() => {
    // Initialize WebSocket connection with session ID
    const wsEndpoint = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/chat';
    wsRef.current = new ChatWebSocket(wsEndpoint, sessionId);
    
    wsRef.current.connect({
      onConnect: () => {
        console.log('Connected to chat WebSocket with session:', sessionId);
        setIsConnected(true);
      },
      onDisconnect: () => {
        console.log('Disconnected from chat WebSocket');
        setIsConnected(false);
      },
      onError: (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      }
    });

    return () => {
      wsRef.current?.disconnect();
    };
  }, [sessionId]);

  const sendMessageToAI = (messageText: string, isFromInput: boolean = true) => {
    if (!wsRef.current?.isConnected || isTyping) return;

    // Add user message immediately only if it's from input (not from error button)
    if (isFromInput) {
      onSendMessage(messageText);
      setInputValue('');
    }
    
    setIsTyping(true);
    setCurrentMessage('');

    // Convert messages to WebSocket format
    const history: WSChatMessage[] = messages.map(msg => ({
      role: msg.role as 'user' | 'assistant',
      content: msg.content
    }));

    // Merge initial codes with edited codes (edited codes override initial)
    const mergedCodes = { ...initialCodes, ...allEditedCodes };
    
    // Send message via WebSocket with cell context and code
    let accumulatedMessage = '';
    wsRef.current.sendMessage(messageText, history, currentCellId, mergedCodes, {
      onProgress: (chunk: string) => {
        accumulatedMessage += chunk;
        setCurrentMessage(accumulatedMessage);
      },
      onEnd: () => {
        // Add the complete assistant message
        if (accumulatedMessage.trim()) {
          onSendMessage(accumulatedMessage, 'assistant');
        }
        setIsTyping(false);
        setCurrentMessage('');
      },
      onError: (error) => {
        console.error('Chat error:', error);
        setIsTyping(false);
        setCurrentMessage('');
        // Add error message
        onSendMessage('Sorry, I encountered an error. Please try again.', 'assistant');
      }
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      sendMessageToAI(inputValue.trim(), true);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.pdf')) {
      setUploadError('Please select a PDF file');
      return;
    }

    await uploadPdf(file);
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const uploadPdf = async (file: File) => {
    setIsUploading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_id', sessionId);

      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/upload_pdf`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to upload PDF');
      }

      const data = await response.json();
      setUploadedPdf({
        name: data.pdf_name,
        numPages: data.num_pages,
        numChunks: data.num_chunks,
      });
      console.log('PDF uploaded successfully:', data);
    } catch (error) {
      console.error('Error uploading PDF:', error);
      setUploadError(error instanceof Error ? error.message : 'Failed to upload PDF');
    } finally {
      setIsUploading(false);
    }
  };

  const handleRemovePdf = () => {
    setUploadedPdf(null);
    setUploadError(null);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const suggestedQuestions = [
    "Explain the gene mapping process",
    "What does the AUC score mean?",
    "How does the GNN model work?",
    "Show me the key Alzheimer's genes",
    "Modify the learning rate"
  ];

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Bot className="w-6 h-6 text-blue-600" />
            <div>
              <h2 className="text-lg font-semibold text-gray-800">AI Assistant</h2>
              <p className="text-sm text-gray-600">OpenAI Streaming Chat</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <Wifi className="w-5 h-5 text-green-500" />
            ) : (
              <WifiOff className="w-5 h-5 text-red-500" />
            )}
            <span className={`text-xs ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        {/* PDF Upload Section */}
        <div className="mt-4 space-y-2">
          {/* Upload Button */}
          {!uploadedPdf && (
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileSelect}
                className="hidden"
                disabled={isUploading}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center space-x-2 px-3 py-2 text-sm bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isUploading ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    <span>Uploading & Indexing...</span>
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    <span>Upload Research Paper (PDF)</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* Uploaded PDF Info */}
          {uploadedPdf && (
            <div className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center space-x-2">
                <FileText className="w-4 h-4 text-green-600" />
                <div>
                  <p className="text-sm font-medium text-green-900">{uploadedPdf.name}</p>
                  <p className="text-xs text-green-700">
                    {uploadedPdf.numPages} pages • {uploadedPdf.numChunks} chunks indexed
                  </p>
                </div>
              </div>
              <button
                onClick={handleRemovePdf}
                className="p-1 hover:bg-green-100 rounded transition-colors"
                title="Remove PDF"
              >
                <X className="w-4 h-4 text-green-700" />
              </button>
            </div>
          )}

          {/* Upload Error */}
          {uploadError && (
            <div className="p-2 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-xs text-red-700">{uploadError}</p>
            </div>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <div className="flex items-start space-x-2">
                {message.role === 'assistant' && (
                  <Bot className="w-4 h-4 mt-0.5 flex-shrink-0 text-blue-600" />
                )}
                <div className="flex-1 min-w-0">
                  {message.role === 'assistant' ? (
                    <MarkdownMessage content={message.content} className="text-sm" />
                  ) : (
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  )}
                  <p className={`text-xs mt-1 ${
                    message.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                  }`}>
                    {formatTime(message.timestamp)}
                  </p>
                </div>
                {message.role === 'user' && (
                  <User className="w-4 h-4 mt-0.5 flex-shrink-0 text-blue-100" />
                )}
              </div>
            </div>
          </div>
        ))}
        
        {/* Streaming message */}
        {isTyping && currentMessage && (
          <div className="flex justify-start">
            <div className="bg-gray-100 text-gray-800 rounded-lg p-3 max-w-[80%]">
              <div className="flex items-start space-x-2">
                <Bot className="w-4 h-4 mt-0.5 flex-shrink-0 text-blue-600" />
                <div className="flex-1 min-w-0">
                  <MarkdownMessage content={currentMessage} className="text-sm" />
                  <div className="inline-block w-2 h-4 bg-blue-600 animate-pulse ml-1"></div>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Typing indicator */}
        {isTyping && !currentMessage && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg p-3 max-w-[80%]">
              <div className="flex items-center space-x-2">
                <Bot className="w-4 h-4 text-blue-600" />
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 p-4">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask about the analysis or request code modifications..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isTyping || !isConnected}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
        <p className="text-xs text-gray-500 mt-2">
          AI can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  );
};