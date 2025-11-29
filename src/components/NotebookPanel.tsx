import React from 'react';
import { Code } from 'lucide-react';
import { PipelineStep, StepResult } from '../types';
import { NotebookView } from './NotebookView';

interface NotebookPanelProps {
  currentStep: PipelineStep | null;
  stepResult: StepResult | null;
  onStepComplete?: (stepId: string, success: boolean) => void;
  onCodeChange?: (code: string) => void; // Add prop for code change callback
  onAllCodesChange?: (allCodes: Record<string, string>) => void; // Add prop for all edited codes
  onSendErrorToChat?: (errorMessage: string) => void; // Add prop for sending errors to chat
  initialCodes?: Record<string, string>;
  currentNotebook?: string; // Add prop to track current notebook
}

export const NotebookPanel: React.FC<NotebookPanelProps> = ({
  currentStep,
  stepResult,
  onStepComplete,
  onCodeChange,
  onAllCodesChange,
  onSendErrorToChat,
  initialCodes,
  currentNotebook
}) => {
  if (!currentStep) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        <div className="text-center">
          <div className="text-6xl mb-4">📊</div>
          <h3 className="text-lg font-medium mb-2">No Step Selected</h3>
          <p>Select a pipeline step from the left panel to view its code and results</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b border-gray-200 bg-white p-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-800">
              {currentStep.title}
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              {currentStep.description}
            </p>
          </div>
          
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <Code className="w-4 h-4" />
            <span>Code View</span>
          </div>
        </div>
      </div>

      {/* Content area */}
      <div className="flex-1 overflow-hidden">
        <NotebookView
          key={currentNotebook} // Force remount when notebook changes
          currentStep={currentStep}
          onStepComplete={onStepComplete}
          onCodeChange={onCodeChange} // Pass the code change handler
          onAllCodesChange={onAllCodesChange} // Pass all codes change handler
          onSendErrorToChat={onSendErrorToChat} // Pass the error sender handler
          initialCodes={initialCodes}
          currentNotebook={currentNotebook} // Pass current notebook for state clearing
        />
      </div>
    </div>
  );
};