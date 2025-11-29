import React, { useEffect, useState, useCallback } from 'react';
import { Sidebar, SidebarSection } from './components/Sidebar';
import { SidePanel } from './components/SidePanel';
import { PipelinePanel } from './components/PipelinePanel';
import { NotebookPanel } from './components/NotebookPanel';
import { ChatPanel } from './components/ChatPanel';
import { PipelineStep, ChatMessage } from './types';

interface Notebook {
  filename: string;
  size: number;
  modified_at: string;
  is_current: boolean;
}

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
  const [allEditedCodes, setAllEditedCodes] = useState<Record<string, string>>({}); // All edited codes
  const [initialCodes, setInitialCodes] = useState<Record<string, string>>({});
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);
  const [currentNotebook, setCurrentNotebook] = useState<string>('colab.ipynb');
  const [notebookVersion, setNotebookVersion] = useState<number>(0); // Track notebook changes
  const [activeSection, setActiveSection] = useState<SidebarSection>('notebooks'); // Sidebar state
  const API_BASE = import.meta.env.VITE_API_BASE || '';

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

  // Function to update all edited codes from NotebookView
  const handleAllCodesChange = (allCodes: Record<string, string>) => {
    setAllEditedCodes(allCodes);
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

  // Notebook management functions
  const fetchNotebooks = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/notebooks`);
      if (!res.ok) throw new Error('Failed to fetch notebooks');
      const data = await res.json();
      setNotebooks(data.notebooks || []);
      setCurrentNotebook(data.current || 'colab.ipynb');
    } catch (error) {
      console.error('Error fetching notebooks:', error);
    }
  }, [API_BASE]);

  const handleUploadNotebook = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/api/notebooks/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Upload failed');
    }

    // After successful upload, refresh notebooks and select the newly uploaded one
    await fetchNotebooks();
    await handleSelectNotebook(file.name);
  };

  const handleSelectNotebook = async (filename: string) => {
    const res = await fetch(`${API_BASE}/api/notebooks/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename }),
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Selection failed');
    }

    // Clear all state before loading new notebook
    setSteps([]);
    setInitialCodes({});
    setCurrentStepId(null);

    // Update current notebook and reload steps
    setCurrentNotebook(filename);
    setNotebookVersion(prev => prev + 1); // Increment version to force remount
    await fetchNotebooks();
    await fetchSteps();
  };

  const handleDeleteNotebook = async (filename: string) => {
    const res = await fetch(`${API_BASE}/api/notebooks/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Delete failed');
    }

    await fetchNotebooks();
  };

  // Load steps dynamically from backend (current notebook)
  const fetchSteps = useCallback(async () => {
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
  }, [API_BASE]);

  // Load notebooks on mount
  useEffect(() => {
    fetchNotebooks();
  }, [fetchNotebooks]);

  // Load steps when component mounts
  useEffect(() => {
    fetchSteps();
  }, [fetchSteps]);

  return (
    <div className="h-screen flex bg-gray-50">
      {/* Left: Icon Sidebar */}
      <Sidebar
        activeSection={activeSection}
        onSectionChange={setActiveSection}
      />

      {/* Left Panel: Collapsible Content */}
      <SidePanel
        activeSection={activeSection}
        onClose={() => setActiveSection(null)}
        notebooks={notebooks}
        currentNotebook={currentNotebook}
        onUploadNotebook={handleUploadNotebook}
        onSelectNotebook={handleSelectNotebook}
        onDeleteNotebook={handleDeleteNotebook}
        onRefreshNotebooks={fetchNotebooks}
      />

      {/* Pipeline Steps Panel */}
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
          key={`${currentNotebook}-${notebookVersion}`} // Force remount on notebook change or version change
          currentStep={currentStepId ? steps.find(s => s.id === currentStepId) || null : null}
          stepResult={null}
          onStepComplete={handleStepComplete}
          onCodeChange={handleCodeChange} // Pass the code change handler
          onAllCodesChange={handleAllCodesChange} // Pass all codes change handler
          onSendErrorToChat={handleSendErrorToChat} // Pass the error sender handler
          initialCodes={initialCodes}
          currentNotebook={currentNotebook} // Pass current notebook for state clearing
        />
      </div>

      {/* Right Panel - Chat */}
      <div className="w-96 border-l border-gray-200 bg-white flex-shrink-0">
        <ChatPanel
          messages={messages}
          onSendMessage={handleSendMessage}
          currentCode={currentCode} // Pass current code to ChatPanel (legacy)
          currentCellId={currentStepId || undefined} // Pass current cell ID
          allEditedCodes={allEditedCodes} // Pass all edited codes
          initialCodes={initialCodes} // Pass initial codes from file
        />
      </div>
    </div>
  );
}

export default App;