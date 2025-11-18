import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Wifi, WifiOff, Upload } from 'lucide-react';
import { ChatMessage } from '../types';
import { ChatWebSocket, ChatMessage as WSChatMessage } from '../lib/chat-websocket';

interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (content: string, role?: 'user' | 'assistant') => void;
  currentCode?: string; // Add prop to receive current cell code
  apiBase?: string;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  messages,
  onSendMessage,
  currentCode,
  apiBase = '',
}) => {
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [currentMessage, setCurrentMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<ChatWebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [isUploadingPlot, setIsUploadingPlot] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

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
    // Initialize WebSocket connection
    const wsEndpoint = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/chat';
    wsRef.current = new ChatWebSocket(wsEndpoint);
    
    wsRef.current.connect({
      onConnect: () => {
        console.log('Connected to chat WebSocket');
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
  }, []);

  const sendMessageToAI = (messageText: string, isFromInput: boolean = true) => {
    if (!wsRef.current?.isConnected || isTyping) return;

    // Create message content that includes current code if available
    let messageContent = messageText;
    if (currentCode && currentCode.trim()) {
      messageContent = `User Message: ${messageText}\n\nCurrent Cell Code:\n\`\`\`python\n${currentCode}\n\`\`\``;
    }
    
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

    // Send message via WebSocket with enhanced content
    let accumulatedMessage = '';
    wsRef.current.sendMessage(messageContent, history, {
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

  const handleUploadButtonClick = () => {
    if (!isUploadingPlot) {
      fileInputRef.current?.click();
    }
  };

  const uploadPlotToBackend = async (file: File) => {
    const baseUrl = apiBase?.replace(/\/$/, '') ?? '';
    const endpoint = `${baseUrl}/api/plots/upload`;
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || 'Failed to upload plot');
    }

    return response.json();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadError(null);
    setIsUploadingPlot(true);

    try {
      await uploadPlotToBackend(file);
      setUploadedFileName(file.name);
    } catch (err) {
      console.error('Plot upload failed', err);
      setUploadError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploadingPlot(false);
      // Allow uploading the same file again if desired
      event.target.value = '';
    }
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
              <div className="mt-2">
                <input
                  ref={fileInputRef}
                  id="plot-upload-input"
                  type="file"
                  accept="image/png"
                  className="hidden"
                  onChange={handleFileChange}
                />
                <button
                  type="button"
                  onClick={handleUploadButtonClick}
                  disabled={isUploadingPlot}
                  className={`inline-flex items-center px-3 py-1 text-sm font-medium text-white rounded-md transition-colors ${
                    isUploadingPlot ? 'bg-blue-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
                  }`}
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Upload plot as png
                </button>
                {uploadedFileName && (
                  <p className="text-xs text-gray-600 mt-1">
                    Uploaded: {uploadedFileName}
                  </p>
                )}
                {uploadError && (
                  <p className="text-xs text-red-500 mt-1">
                    {uploadError}
                  </p>
                )}
              </div>
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
                  <Bot className="w-4 h-4 mt-0.5 text-blue-600" />
                )}
                <div className="flex-1">
                  <p className="text-sm">{message.content}</p>
                  <p className={`text-xs mt-1 ${
                    message.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                  }`}>
                    {formatTime(message.timestamp)}
                  </p>
                </div>
                {message.role === 'user' && (
                  <User className="w-4 h-4 mt-0.5 text-blue-100" />
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
                <Bot className="w-4 h-4 mt-0.5 text-blue-600" />
                <div className="flex-1">
                  <p className="text-sm whitespace-pre-wrap">{currentMessage}</p>
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