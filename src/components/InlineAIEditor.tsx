import React, { useState, useEffect, useRef } from 'react';
import { Sparkles, X } from 'lucide-react';

interface InlineAIEditorProps {
  onSendQuery: (query: string, selectedText: string) => void;
  isVisible: boolean;
  onClose: () => void;
  selectedText: string;
  position: { x: number; y: number };
}

export const InlineAIEditor: React.FC<InlineAIEditorProps> = ({
  onSendQuery,
  isVisible,
  onClose,
  selectedText,
  position
}) => {
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when editor opens
  useEffect(() => {
    if (isVisible && inputRef.current) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 10);
    }
  }, [isVisible]);

  // Reset query when editor closes
  useEffect(() => {
    if (!isVisible) {
      setQuery('');
    }
  }, [isVisible]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSendQuery(query, selectedText);
      setQuery('');
      onClose();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isVisible) return null;

  return (
    <div
      className="fixed z-50"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
      }}
    >
      <div className="bg-white border border-blue-500 rounded-lg shadow-lg min-w-[400px] max-w-[600px]">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 bg-blue-50">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-blue-600" />
            <span className="font-medium text-blue-900">Ask AI about selected code</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-blue-100 rounded transition-colors"
          >
            <X className="w-4 h-4 text-blue-600" />
          </button>
        </div>

        {/* Context Preview */}
        <div className="px-4 py-2 bg-gray-50 border-b border-gray-200">
          <div className="text-xs text-gray-600 mb-1">Selected code:</div>
          <div className="text-xs font-mono bg-gray-100 p-2 rounded max-h-24 overflow-y-auto">
            {selectedText.substring(0, 200)}{selectedText.length > 200 ? '...' : ''}
          </div>
        </div>

        {/* Query Input */}
        <form onSubmit={handleSubmit} className="p-4">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about the selected code..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <div className="flex items-center justify-between mt-2">
            <span className="text-xs text-gray-500">
              Press Enter to send, Esc to cancel
            </span>
            <button
              type="submit"
              disabled={!query.trim()}
              className="px-4 py-1.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors text-sm font-medium"
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

