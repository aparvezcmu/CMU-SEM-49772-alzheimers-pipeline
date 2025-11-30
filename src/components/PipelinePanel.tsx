import React from 'react';
import { CheckCircle, Clock, AlertCircle } from 'lucide-react';
import { PipelineStep } from '../types';
import clsx from 'clsx';

interface PipelinePanelProps {
  steps: PipelineStep[];
  currentStepId: string | null;
  onStepClick: (stepId: string) => void;
}

export const PipelinePanel: React.FC<PipelinePanelProps> = ({
  steps,
  currentStepId,
  onStepClick,
}) => {
  const getStatusIcon = (status: PipelineStep['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'running':
        return <Clock className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <div className="w-5 h-5 rounded-full border-2 border-gray-300" />;
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-800">
          Pipeline Steps
        </h2>
        <p className="text-sm text-gray-600 mt-1">
          Click steps to view and execute code
        </p>
      </div>

      <div className="flex-1 overflow-y-auto">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={clsx(
              'step-item',
              currentStepId === step.id && 'step-active'
            )}
            onClick={() => onStepClick(step.id)}
          >
            <div className="flex items-center justify-between w-full">
              <div className="flex items-center space-x-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gray-100 text-sm font-semibold text-gray-600">
                  {index + 1}
                </div>
                <div className="flex-1">
                  <h3 className="font-medium text-gray-800">{step.title}</h3>
                  <p className="text-sm text-gray-600">{step.description}</p>
                  {step.executionCount && (
                    <span className="execution-count">
                      [Executed {step.executionCount} time{step.executionCount > 1 ? 's' : ''}]
                    </span>
                  )}
                </div>
              </div>

              <div className="flex items-center space-x-2">
                {getStatusIcon(step.status)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
