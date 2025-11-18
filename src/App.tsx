import React, { useEffect, useState } from 'react';
import { PipelinePanel } from './components/PipelinePanel';
import { NotebookPanel } from './components/NotebookPanel';
import { ChatPanel } from './components/ChatPanel';
import { PipelineStep, ChatMessage } from './types';

// Steps will be loaded dynamically from the notebook
const initialSteps: PipelineStep[] = [];

const initialMessages: ChatMessage[] = [
  {
    id: '1',
    content: 'Hello! I\'m your AI assistant for Alzheimer\'s disease research. I can help you understand the pipeline, modify code, or answer questions about the analysis.',
    role: 'assistant',
    timestamp: new Date()
  }
];

function App() {
  const [steps, setSteps] = useState<PipelineStep[]>(initialSteps);
  const [currentStepId, setCurrentStepId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [currentCode, setCurrentCode] = useState<string>(''); // Add state for current cell code
  const [initialCodes, setInitialCodes] = useState<Record<string, string>>({});
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

  const handleStepClick = (stepId: string) => {
    setCurrentStepId(stepId);
  };

  const handleStepComplete = (stepId: string, success: boolean) => {
    console.log(`Step ${stepId} ${success ? 'completed successfully' : 'failed'}`);
    
    // Update step status based on success/failure
    setSteps(prev => prev.map(step => 
      step.id === stepId 
        ? { 
            ...step, 
            status: success ? 'completed' : 'error',
            executionCount: (step.executionCount || 0) + 1
          }
        : step
    ));
  };

  const handleSendMessage = (content: string, role: 'user' | 'assistant' = 'user') => {
    const message: ChatMessage = {
      id: Date.now().toString(),
      content,
      role,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, message]);
  };

  // Function to update current code from NotebookView
  const handleCodeChange = (code: string) => {
    setCurrentCode(code);
  };

  // Function to send error message to chat
  const handleSendErrorToChat = (errorMessage: string) => {
    // First add the user message to the chat
    handleSendMessage(errorMessage);
    
    // Then trigger the ChatPanel to send this message to the AI
    // We need to find a way to programmatically trigger the send
    // For now, let's add a small delay and then trigger the input submission
    setTimeout(() => {
      const event = new CustomEvent('sendErrorMessage', { 
        detail: { message: errorMessage, code: currentCode } 
      });
      window.dispatchEvent(event);
    }, 100);
  };

  // Load steps dynamically from backend (colab.ipynb)
  useEffect(() => {
    const fetchSteps = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/notebook/cells`);
        if (!res.ok) throw new Error('Failed to load notebook steps');
        const data = await res.json();
        const loadedSteps: PipelineStep[] = (data.steps || []).map((s: any) => ({
          id: `step-${s.stepNumber}`,
          title: s.title,
          description: s.description,
          status: 'pending',
          notebookCellIndex: s.index,
        }));
        setSteps(loadedSteps);
        if (loadedSteps.length > 0) {
          setCurrentStepId(loadedSteps[0].id);
        }

        // Preload all codes once
        const codeEntries: [string, string][] = await Promise.all(
          loadedSteps.map(async (st) => {
            if (st.notebookCellIndex === undefined) return [st.id, ''];
            try {
              const r = await fetch(`${API_BASE}/api/notebook/cell/${st.notebookCellIndex}`);
              if (!r.ok) return [st.id, ''];
              const j = await r.json();
              return [st.id, j.source || ''];
            } catch {
              return [st.id, ''];
            }
          })
        );
        setInitialCodes(Object.fromEntries(codeEntries));
      } catch (e) {
        console.error('Error loading steps from notebook', e);
      }
    };
    fetchSteps();
  }, []);

  return (
    <div className="h-screen flex bg-gray-50">
      {/* Left Panel - Pipeline Steps */}
      <div className="w-80 border-r border-gray-200 bg-white flex-shrink-0">
        <PipelinePanel 
          steps={steps}
          currentStepId={currentStepId}
          onStepClick={handleStepClick}
        />
      </div>

      {/* Center Panel - Code Execution */}
      <div className="flex-1 flex flex-col min-w-0">
        <NotebookPanel
          currentStep={currentStepId ? steps.find(s => s.id === currentStepId) || null : null}
          stepResult={null}
          onStepComplete={handleStepComplete}
          onCodeChange={handleCodeChange} // Pass the code change handler
          onSendErrorToChat={handleSendErrorToChat} // Pass the error sender handler
          initialCodes={initialCodes}
        />
      </div>

      {/* Right Panel - Chat */}
      <div className="w-96 border-l border-gray-200 bg-white flex-shrink-0">
        <ChatPanel
          messages={messages}
          onSendMessage={handleSendMessage}
          currentCode={currentCode} // Pass current code to ChatPanel
          apiBase={API_BASE}
        />
      </div>
    </div>
  );
}

export default App;