import React, { useState, useEffect, useRef } from 'react';
import { CellOutput, PipelineStep } from '../types';
import { RotateCcw, AlertCircle } from 'lucide-react';
import { InlineAIEditor } from './InlineAIEditor';

// Define code templates corresponding to each pipeline step
const stepCodeTemplates: Record<string, string> = {
  'step-1': `# Load Dataset - Download and load Alzheimer's disease dataset
import pandas as pd
import numpy as np
import scanpy as sc

print("Loading single-cell transcriptomics excitatory data...")

# Simulate loading dataset
dataset = pd.DataFrame({
    'cell_id': range(46070),
    'gene_count': np.random.poisson(1000, 46070),
    'condition': np.random.choice(['Early AD', 'Normal'], 46070)
})

print("Dataset shape: (46070, 33091)")
print("Genes: 33,091")
print("Cells: 46,070")
print("Successfully loaded dataset")
print("• Total cells: 46,070")
print("• Total genes: 33,091")
print("• Cell types: Ex01_CUX2-LAMP5 (L2-L3), Ex02_CUX2-COL5A2 (L2-L4)")
print("• Conditions: Early AD, Normal")`,

  'step-2': `# Gene Mapping - Map gene IDs to symbols using MyGene API
# Check if dataset from step 1 exists
try:
    dataset
    print("Found dataset from previous step")
except NameError:
    print("ERROR: Dataset not found!")
    print("Please run Step 1 first to load the dataset.")
    raise NameError("Dataset variable not found. Run Step 1 first.")

print("Mapping 33,091 gene IDs...")

# Example gene mappings
gene_mappings = {
    'ENSG00000223554': 'DDX11L1',
    'ENSG00000175315': 'WASH7P',
    'ENSG00000139800': 'HECTD3',
    'ENSG00000265163': 'MIR6859-1',
    'ENSG00000184856': 'CDKN2B-AS1'
}

print(f"Successfully mapped {len(gene_mappings)} genes")
print("Example mappings:")
for ensembl, symbol in gene_mappings.items():
    print(f"  {ensembl} -> {symbol}")

print("\\nGene mapping completed")
print("• Success rate: 99.99% (33,087/33,091)")
print("• Failed mappings: 4 genes")
print("• Key AD genes found: APOE, APP, PSEN1, PSEN2, TREM2")`,

  'step-3': `# Data Preprocessing - Normalize and filter gene expression data
# Check if required variables exist
try:
    dataset
    gene_mappings
    print("Found required variables from previous steps")
except NameError as e:
    print("ERROR: Missing required variables!")
    print("Please run previous steps first.")
    raise NameError(f"Missing variables: {e}")

print("Starting data preprocessing...")

# Normalization steps
print("Normalizing to 10,000 reads per cell...")
print("Applying log transformation: log(x+1)")
print("Filtering cells and genes...")

# Show results
print("Filtered dataset shape: (46070, 28358)")
print("Highly variable genes: 2,847")

print("\\nData Preprocessing Results")
print("• Normalization: 10,000 reads per cell")
print("• Transformation: log(x+1) applied")
print("• Cell filtering: removed 0 cells (min 200 genes)")
print("• Gene filtering: removed 4,733 genes (min 3 cells)")
print("• Highly variable genes: 2,847 identified")
print("• Final dataset: 46,070 cells × 28,358 genes")`,

  'step-4': `# Model Training - Train GNN model with reinforcement learning
# Check if required variables exist
try:
    dataset
    gene_mappings
    print("Found required variables from previous steps")
except NameError as e:
    print("ERROR: Missing required variables!")
    print("Please run previous steps first.")
    raise NameError(f"Missing variables: {e}")

import matplotlib.pyplot as plt
import time

print("Training GNN model...")
print("Architecture: 2-layer GCN + MLP")
print("Parameters: 1,814,658 trainable")

# Simulate training
losses = []
for epoch in range(0, 10, 2):
    loss = 0.6931 - (epoch * 0.05)
    losses.append(loss)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Model training completed!")

# Plot training curve
plt.figure(figsize=(8, 5))
plt.plot([0, 2, 4, 6, 8], losses, 'b-o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GNN Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.show()

print("\\nGNN Training Results")
print("• Architecture: 2-layer GCN + MLP")
print("• Parameters: 1,814,658 trainable")
print("• Training time: 45.2 seconds")
print("• Final loss: 0.245")
print("• Convergence: achieved at epoch 8")`,

  'step-5': `# Performance Metrics - Evaluate model performance and generate plots
# Check if required variables exist
try:
    dataset
    gene_mappings
    print("Found required variables from previous steps")
except NameError as e:
    print("ERROR: Missing required variables!")
    print("Please run previous steps first.")
    raise NameError(f"Missing variables: {e}")

import matplotlib.pyplot as plt
import numpy as np

print("Evaluating model performance...")

# Simulate performance metrics
accuracy = 0.92
precision = 0.89
recall = 0.94
f1_score = 0.91
auc_roc = 0.95

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1_score:.2%}")
print(f"AUROC: {auc_roc:.2%}")

# Create performance visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix
conf_matrix = np.array([[850, 50], [70, 930]])
im = ax1.imshow(conf_matrix, cmap='Blues', alpha=0.8)
ax1.set_title('Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Normal', 'AD'])
ax1.set_yticklabels(['Normal', 'AD'])

# Add text annotations
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(conf_matrix[i, j]), 
                ha='center', va='center', fontsize=14, fontweight='bold')

# ROC curve
fpr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
tpr = np.array([0, 0.85, 0.88, 0.91, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 1.0])
ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_roc:.2f})')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\nPerformance Evaluation Results")
print("• Accuracy: 92.0%")
print("• Precision: 89.0%")
print("• Recall: 94.0%")
print("• F1-Score: 91.0%")
print("• AUROC: 95.0%")
print("• Model successfully distinguishes AD vs Normal cells")`
};

interface NotebookViewProps {
  currentStep: PipelineStep | null;
  onStepComplete?: (stepId: string, success: boolean) => void;
  onCodeChange?: (code: string) => void; // Add prop for code change callback
  onAllCodesChange?: (allCodes: Record<string, string>) => void; // Add prop for all edited codes
  onSendErrorToChat?: (errorMessage: string) => void; // Add prop for sending errors to chat
  initialCodes?: Record<string, string>;
  currentNotebook?: string; // Add prop to track current notebook
}


export const NotebookView: React.FC<NotebookViewProps> = ({ currentStep, onStepComplete, onCodeChange, onAllCodesChange, onSendErrorToChat, initialCodes, currentNotebook }) => {
  const [cellStates, setCellStates] = useState<Record<string, {
    executed: boolean;
    executing: boolean;
    outputs: CellOutput[];
    executionTime?: number; // Add execution time tracking
  }>>({});

  const [editableCode, setEditableCode] = useState<Record<string, string>>({});
  // Terminal streaming removed; keep simple aggregated output

  // Kernel restart states
  const [isRestartingKernel, setIsRestartingKernel] = useState(false);
  const [showRestartConfirmation, setShowRestartConfirmation] = useState(false);

  // Inline AI Editor states
  const [showInlineAIEditor, setShowInlineAIEditor] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [editorPosition, setEditorPosition] = useState({ x: 0, y: 0 });
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Get current step's cell state
  const getCurrentCellState = () => {
    if (!currentStep) return { executed: false, executing: false, outputs: [] };
    return cellStates[currentStep.id] || { executed: false, executing: false, outputs: [] };
  };

  const currentCellState = getCurrentCellState();

  // Get current step's code (either edited or from initialCodes)
  const getCurrentCode = () => {
    if (!currentStep) return '';
    // Return edited code if exists, otherwise return from initialCodes
    return editableCode[currentStep.id] || (initialCodes && initialCodes[currentStep.id]) || '';
  };

  // Check if current code has been edited
  const isCodeEdited = () => {
    if (!currentStep) return false;
    return editableCode[currentStep.id] !== undefined;
  };

  // Update current code when it changes
  useEffect(() => {
    const currentCode = getCurrentCode();
    if (onCodeChange && currentCode) {
      onCodeChange(currentCode);
    }
  }, [currentStep, editableCode, onCodeChange]);

  // Update all codes when editableCode changes
  useEffect(() => {
    if (onAllCodesChange) {
      onAllCodesChange(editableCode);
    }
  }, [editableCode, onAllCodesChange]);

  const handleExplainError = (errorOutput: CellOutput) => {
    if (!onSendErrorToChat || !currentStep) {
      return;
    }
    
    // Simplified error message - just ask for help with the error type
    const errorMessage = `Please explain this error: ${errorOutput.ename}: ${errorOutput.evalue}`;
    
    onSendErrorToChat(errorMessage);
  };

  const executeCode = async (stepId?: string) => {
    const targetStepId = stepId || currentStep?.id;
    if (!targetStepId) return;

    const code = getCurrentCode();
    if (!code) {
      console.error(`No code found for step ${targetStepId}`);
      return;
    }

    // Start timing
    const startTime = Date.now();

    // Update executing state for the specific step
    setCellStates(prev => ({
      ...prev,
      [targetStepId]: {
        ...prev[targetStepId],
        executing: true,
        outputs: []
      }
    }));

    try {
      const API_BASE = import.meta.env.VITE_API_BASE || '';
      const response = await fetch(`${API_BASE}/api/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code: code,
          cell_id: (currentStep && currentStep.notebookCellIndex !== undefined)
            ? currentStep.notebookCellIndex
            : parseInt(targetStepId.split('-')[1])
        })
      });

      // Calculate execution time
      const endTime = Date.now();
      const executionTime = (endTime - startTime) / 1000; // Convert to seconds

      if (response.ok) {
        const result = await response.json();
        
        // Check if execution was successful
        const hasError = result.outputs && result.outputs.some((output: any) => output.type === 'error');
        
        // Update cell state for the specific step with execution time
        setCellStates(prev => ({
          ...prev,
          [targetStepId]: {
            executed: result.status === 'ok' && !hasError,
            executing: false,
            outputs: result.outputs || [],
            executionTime: executionTime
          }
        }));

        // Notify parent component about step completion (success or failure)
        if (onStepComplete) {
          onStepComplete(targetStepId, result.status === 'ok' && !hasError);
        }

      } else {
        throw new Error('Execution failed');
      }
    } catch (error) {
      // Calculate execution time even for errors
      const endTime = Date.now();
      const executionTime = (endTime - startTime) / 1000;

      console.error('Error executing code:', error);
      setCellStates(prev => ({
        ...prev,
        [targetStepId]: {
          executed: false,
          executing: false,
          outputs: [{
            type: 'error',
            content: `Error: ${error instanceof Error ? error.message : String(error)}`,
            ename: 'ExecutionError',
            evalue: error instanceof Error ? error.message : String(error),
            traceback: [error instanceof Error ? error.message : String(error)]
          }],
          executionTime: executionTime
        }
      }));
      
      // Notify parent component about step failure
      if (onStepComplete) {
        onStepComplete(targetStepId, false);
      }
    }
  };


  const runCurrentStep = async () => {
    if (!currentStep) return;
    await executeCode(currentStep.id);
  };

  // Initialize code from preloaded initialCodes when step changes
  useEffect(() => {
    if (!currentStep) return;
    if (!initialCodes) return;
    const preloaded = initialCodes[currentStep.id];
    if (preloaded === undefined) return;

    // Only set the code if it hasn't been edited yet
    setEditableCode(prev => {
      // If this step already has edited code, don't overwrite it
      if (prev[currentStep.id] !== undefined) {
        return prev;
      }
      return {
        ...prev,
        [currentStep.id]: preloaded
      };
    });
  }, [currentStep, initialCodes]);

  const handleCodeChange = (newCode: string) => {
    if (!currentStep) return;
    setEditableCode(prev => ({
      ...prev,
      [currentStep.id]: newCode
    }));

    // Reset execution state and time when code is modified
    setCellStates(prev => ({
      ...prev,
      [currentStep.id]: {
        ...prev[currentStep.id],
        executed: false,
        outputs: [],
        executionTime: undefined // Reset execution time
      }
    }));
  };

  // Handle kernel restart
  const handleRestartKernel = async () => {
    setShowRestartConfirmation(false);
    setIsRestartingKernel(true);

    try {
      const API_BASE = import.meta.env.VITE_API_BASE || '';
      const response = await fetch(`${API_BASE}/api/restart_kernel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const result = await response.json();
        if (result.status === 'restarted') {
          // Clear all cell states after kernel restart
          setCellStates({});

          // Show success message in the current step's output
          if (currentStep) {
            setCellStates(prev => ({
              ...prev,
              [currentStep.id]: {
                executed: false,
                executing: false,
                outputs: [{
                  type: 'stream',
                  content: '✓ Kernel restarted successfully.\n\nAll variables and imports have been cleared.\nYou may need to re-run previous cells.',
                  name: 'stdout'
                }],
                executionTime: undefined
              }
            }));
          }
        }
      } else {
        throw new Error('Failed to restart kernel');
      }
    } catch (error) {
      console.error('Error restarting kernel:', error);
      // Show error message
      if (currentStep) {
        setCellStates(prev => ({
          ...prev,
          [currentStep.id]: {
            executed: false,
            executing: false,
            outputs: [{
              type: 'error',
              content: `Failed to restart kernel: ${error instanceof Error ? error.message : String(error)}`,
              ename: 'KernelRestartError',
              evalue: error instanceof Error ? error.message : String(error),
              traceback: []
            }],
            executionTime: undefined
          }
        }));
      }
    } finally {
      setIsRestartingKernel(false);
    }
  };

  // Inline AI Editor handlers
  const handleSelectionChange = () => {
    if (!textareaRef.current) return;
    const textarea = textareaRef.current;
    const selection = textarea.value.substring(textarea.selectionStart, textarea.selectionEnd);
    setSelectedText(selection);
  };

  const handleKeyDownWithSelection = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Handle Ctrl+Enter for code execution
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      runCurrentStep();
      return;
    }

    // Handle Cmd/Ctrl+K for inline AI
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      const textarea = textareaRef.current;
      if (textarea) {
        const selection = textarea.value.substring(textarea.selectionStart, textarea.selectionEnd);
        if (selection.trim()) {
          setSelectedText(selection);
          // Calculate position for the editor popup
          const rect = textarea.getBoundingClientRect();
          setEditorPosition({
            x: rect.left + 20,
            y: rect.top + textarea.scrollTop + 20
          });
          setShowInlineAIEditor(true);
        }
      }
      return;
    }

    // Handle Escape to close editor
    if (e.key === 'Escape' && showInlineAIEditor) {
      setShowInlineAIEditor(false);
    }
  };

  const handleSendInlineQuery = (query: string, context: string) => {
    if (onSendErrorToChat) {
      const fullPrompt = `${query}\n\nContext:\n${context}`;
      onSendErrorToChat(fullPrompt);
    }
  };

  const renderOutput = (output: CellOutput) => {
    switch (output.type) {
      case 'stream':
      case 'text':
        return (
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto text-sm whitespace-pre-wrap">
            {output.content}
          </pre>
        );
      
      case 'error':
        return (
          <div className="bg-red-50 border border-red-200 text-red-800 p-4 rounded-md">
            <div className="font-medium text-red-900 mb-2">
              {output.ename}: {output.evalue}
            </div>
            {output.traceback && (
              <pre className="text-sm overflow-x-auto text-red-700 mb-3">
                {output.traceback.join('\n')}
              </pre>
            )}
            {onSendErrorToChat && (
              <div className="mt-3 pt-3 border-t border-red-200">
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    handleExplainError(output);
                  }}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded-md transition-colors shadow-sm cursor-pointer"
                  type="button"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Explain this error
                </button>
              </div>
            )}
          </div>
        );
      
      case 'image':
        return (
          <div className="bg-white p-4 rounded-md text-center">
            <img 
              src={`data:image/${output.format || 'png'};base64,${output.content}`}
              alt="Plot output" 
              className="max-w-full h-auto mx-auto"
            />
          </div>
        );
      
      case 'html':
        return (
          <div 
            className="bg-white p-4 rounded-md border"
            dangerouslySetInnerHTML={{ __html: output.content }}
          />
        );
      
      default:
        return (
          <pre className="bg-gray-100 p-4 rounded-md text-sm">
            {JSON.stringify(output, null, 2)}
          </pre>
        );
    }
  };

  // Build aggregated console text from outputs
  // Terminal view removed

  if (!currentStep) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500 bg-gray-50">
        <div className="text-center">
          <div className="text-6xl mb-4">📊</div>
          <h3 className="text-lg font-medium mb-2">No Step Selected</h3>
          <p>Select a pipeline step from the left panel to view and execute its code</p>
        </div>
      </div>
    );
  }

  const code = getCurrentCode();

  return (
    <div className="h-full overflow-y-auto bg-gray-50 p-2">
      {/* Restart Kernel Confirmation Dialog */}
      {showRestartConfirmation && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <div className="flex items-start space-x-3 mb-4">
              <AlertCircle className="w-6 h-6 text-yellow-500 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Restart Kernel?
                </h3>
                <p className="text-sm text-gray-600">
                  This will restart the Python kernel and clear all variables, imports, and state.
                </p>
                <p className="text-sm text-gray-600 mt-2">
                  You will need to re-run any cells to restore the previous state.
                </p>
              </div>
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowRestartConfirmation(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleRestartKernel}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-md transition-colors"
              >
                Restart Kernel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto">
        {/* Cell for current step */}
        <div className="bg-white rounded-lg mb-4 overflow-hidden shadow-sm">
          {/* Cell Header */}
          <div className="flex items-center justify-between bg-gray-50 px-4 py-3 border-b border-gray-200">
            <div className="flex items-center space-x-3">
              <span className="text-sm font-medium text-gray-700">
                [{currentStep.id}] {currentStep.title}
              </span>
              {currentCellState.executed && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  ✓ Executed
                </span>
              )}
              {currentCellState.outputs.some(output => output.type === 'error') && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                  ✗ Error
                </span>
              )}
              {currentCellState.executionTime !== undefined && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  ⏱️ {currentCellState.executionTime.toFixed(2)}s
                </span>
              )}
            </div>

            <div className="flex items-center space-x-2">
              {/* Restart Kernel Button */}
              <button
                onClick={() => setShowRestartConfirmation(true)}
                disabled={isRestartingKernel}
                className={`
                  flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium
                  transition-colors duration-200
                  ${isRestartingKernel ?
                    'bg-gray-100 text-gray-500 cursor-not-allowed' :
                    'bg-gray-100 text-gray-700 hover:bg-gray-200 active:bg-gray-300'
                  }
                `}
                title="Restart Kernel"
              >
                <RotateCcw className={`w-4 h-4 ${isRestartingKernel ? 'animate-spin' : ''}`} />
                <span>Restart</span>
              </button>

              {/* Run Code Button */}
              <button
                onClick={runCurrentStep}
                disabled={currentCellState.executing || isRestartingKernel}
                className={`
                  flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium
                  transition-colors duration-200
                  ${(currentCellState.executing || isRestartingKernel) ?
                    'bg-blue-100 text-blue-700 cursor-not-allowed' :
                    'bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800'
                  }
                `}
              >
                {(currentCellState.executing) ? (
                  <>
                    <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                    <span>Running...</span>
                  </>
                ) : (
                  <>
                    <div className="w-4 h-4">▶</div>
                    <span>Run Code</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Code Content */}
          <div className="p-4">
            <div className="relative">
              {isCodeEdited() && (
                <div className="mb-2 flex items-center space-x-2">
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                    ✏️ Modified
                  </span>
                </div>
              )}
              
              <textarea
                ref={textareaRef}
                value={code}
                onChange={(e) => handleCodeChange(e.target.value)}
                onSelect={handleSelectionChange}
                onKeyDown={handleKeyDownWithSelection}
                className="w-full h-[500px] bg-gray-50 p-4 rounded-md font-mono text-sm text-gray-800 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 leading-relaxed"
                placeholder="Write any Python code here... (Ctrl+Enter to run, Cmd/Ctrl+K to ask AI about selected code)

Examples:
- import matplotlib.pyplot as plt
- plt.plot([1,2,3,4])
- plt.show()  # Images will display automatically
- import pandas as pd
- import numpy as np"
                spellCheck={false}
                style={{
                  fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace',
                  lineHeight: '1.5',
                  tabSize: 4
                }}
              />
            </div>
          </div>

          {/* Output Area */}
          <div className="border-t border-gray-200 p-4 bg-gray-50">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium text-gray-600">Output:</span>
              {currentCellState.executionTime !== undefined && (
                <span className="text-xs text-gray-500">
                  Execution time: {currentCellState.executionTime.toFixed(2)} seconds
                </span>
              )}
            </div>
            {currentCellState.outputs.length > 0 && (
              <div className="space-y-3">
                {currentCellState.outputs.map((output, index) => (
                  <div key={index}>
                    {renderOutput(output)}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Inline AI Editor */}
      <InlineAIEditor
        isVisible={showInlineAIEditor}
        onClose={() => setShowInlineAIEditor(false)}
        selectedText={selectedText}
        position={editorPosition}
        onSendQuery={handleSendInlineQuery}
      />
    </div>
  );
};