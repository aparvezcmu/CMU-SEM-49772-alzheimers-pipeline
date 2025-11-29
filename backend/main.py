# Load environment variables from .env file FIRST, before any imports
import os
import shutil
from datetime import datetime
import pickle
import tempfile
import uuid
import shutil
from rag_indexer import extract_text_from_pdf, chunk_text, embed_texts, build_faiss_index, retrieve

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, manually load .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio

from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from models.openai_chat import get_openai_streaming_response, format_messages, query_openai_api
from kernel_manager import get_kernel_manager
from storage.azure_blob_manager import get_blob_manager

app = FastAPI(title="Alzheimer's Analysis Pipeline API")

# Global state for current notebook
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "notebooks")  # Local cache directory
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")  # Workspace for data files
CURRENT_NOTEBOOK = "colab.ipynb"  # Default notebook

# Ensure workspace directory exists
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# Initialize Azure Blob Manager
try:
    blob_manager = get_blob_manager()
    USE_AZURE_STORAGE = True
    print("✓ Azure Blob Storage initialized successfully")

    # Upload default notebook to Azure if it exists locally but not in Azure
    default_notebook_path = os.path.join(NOTEBOOKS_DIR, CURRENT_NOTEBOOK)
    if os.path.exists(default_notebook_path) and not blob_manager.notebook_exists(CURRENT_NOTEBOOK):
        try:
            with open(default_notebook_path, "rb") as f:
                content = f.read()
            blob_manager.upload_notebook(CURRENT_NOTEBOOK, content)
            print(f"✓ Uploaded default notebook '{CURRENT_NOTEBOOK}' to Azure Blob Storage")
        except Exception as upload_error:
            print(f"⚠ Warning: Could not upload default notebook to Azure: {upload_error}")

except Exception as e:
    blob_manager = None
    USE_AZURE_STORAGE = False
    print(f"✗ Azure Blob Storage not available: {e}")
    print("  Falling back to local storage")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session-based storage for PDFs and FAISS indexes
# Structure: {session_id: {pdf_path, pdf_name, chunks, embeddings, faiss_index}}
session_storage: Dict[str, Dict[str, Any]] = {}

# Temporary directory for PDF uploads (cleared on restart)
TEMP_DIR = tempfile.mkdtemp(prefix="gene_ide_pdfs_")

# Query context types
class QueryContext(str, Enum):
    DOCUMENT = "document"  # Query is about the uploaded PDF/paper
    CODE = "code"  # Query is about code in the IDE/notebook
    BOTH = "both"  # Query involves both document and code
    GENERAL = "general"  # General conversation, not specific to either

# PDF summary response format
PDF_SUMMARY_FORMAT = """
Format the response using markdown with the following sections:

## Summary

### 1. Prevalence of the Issue
Describe the scope and importance of the problem being addressed.

### 2. Impact & Innovation
Explain the paper's contribution and innovative aspects.

### 3. Existing Landscape
Summarize the current state of research in this area.

### 4. Hypothesis
State the paper's main hypothesis or research question.

### 5. Datasets
List the datasets used (specify if open, closed, or proprietary).

### 6. Methods & Approach
Describe the methodology (e.g., deep learning, machine learning, statistics).
Use **bold** for key techniques and `code` for specific algorithms or frameworks.

### 7. Results
Summarize the main findings and their significance.
Use tables or bullet points for quantitative results when appropriate.

"""

def cleanup_session(session_id: str):
    """Clean up session data including PDF file and FAISS index"""
    if session_id in session_storage:
        session_data = session_storage[session_id]
        # Remove PDF file if it exists
        if 'pdf_path' in session_data and os.path.exists(session_data['pdf_path']):
            try:
                os.remove(session_data['pdf_path'])
            except Exception as e:
                print(f"Error removing PDF file: {e}")
        # Remove from session storage
        del session_storage[session_id]
        print(f"Cleaned up session: {session_id}")

def is_document_summary_query(query: str) -> bool:
    """
    Uses LLM to intelligently detect if a query is asking for a comprehensive 
    paper summary or general explanation (vs. specific section/detail queries).
    
    Returns True if the query requests:
    - Full paper summary/overview
    - General explanation of the entire paper
    - Comprehensive breakdown of the document
    - "What is this paper about" type queries
    
    Returns False for:
    - Specific section queries (methods, results, etc.)
    - Detailed questions about particular aspects
    - Technical clarifications
    """
    instruction = """You are a query classifier for a research paper assistant. 
    
Your task: Determine if the user is asking for a COMPREHENSIVE FULL PAPER SUMMARY/EXPLANATION.

Return "YES" if the query is asking for:
- A complete overview/summary of the entire paper
- General explanation of what the paper is about
- High-level breakdown covering all major aspects (hypothesis, methods, results, impact, etc.)
- "Tell me about this paper" or "explain this paper" type requests
- Full context or broad understanding of the document

Return "NO" if the query is asking for:
- Specific sections only (e.g., just methods, just results, just introduction)
- Detailed technical questions about one aspect
- Clarification of specific terms, figures, or tables
- Comparison or analysis of particular elements
- Questions that are narrow in scope

Examples of YES queries:
- "Can you summarize this paper?"
- "What is this paper about?"
- "Explain the paper to me"
- "Give me an overview"
- "Walk me through this document"
- "What does this paper discuss?"

Examples of NO queries:
- "What methods did they use?"
- "Explain the results section"
- "What is the accuracy of their model?"
- "What dataset did they use?"
- "How does Figure 3 work?"

Respond with ONLY "YES" or "NO"."""

    try:
        response = query_openai_api(instruction, query)
        response_lower = response.strip().lower()
        return "yes" in response_lower and "no" not in response_lower
    except Exception as e:
        print(f"Error in summary query detection: {e}")
        # Fallback: simple heuristic
        query_lower = query.lower()
        summary_keywords = ['summary', 'summarize', 'overview', 'explain the paper', 
                           'explain this paper', 'what is this paper about', 
                           'tell me about', 'describe the paper']
        return any(keyword in query_lower for keyword in summary_keywords)

def classify_query_context(query: str) -> QueryContext:
    """
    Uses LLM to intelligently classify the context of a user query.
    
    Returns:
        QueryContext.DOCUMENT - Query is about the uploaded PDF/research paper
        QueryContext.CODE - Query is about code in the IDE/notebook
        QueryContext.BOTH - Query involves both document and code context
        QueryContext.GENERAL - General conversation, not specific to either
    """
    instruction = """You are a context classifier for an AI research assistant that helps with both research papers and code.

Your task: Classify the user's query into ONE of these categories:

1. DOCUMENT - The query is asking about a research paper, PDF, or document content:
   - Questions about papers, studies, research, findings
   - Asking about methods, results, hypotheses in papers
   - Questions about authors, publications, citations
   - Document summaries, explanations, or specific sections
   Examples: "What does the paper say about...", "Summarize the results", "What methods did the authors use?"

2. CODE - The query is asking about programming, code, or implementation:
   - Questions about code functionality, debugging, or errors
   - Asking to write, modify, or explain code
   - Questions about programming concepts, libraries, or frameworks
   - IDE-related questions about notebooks, cells, or execution
   Examples: "Fix this bug", "Write a function to...", "Why is my code not working?", "Explain this code"

3. BOTH - The query requires BOTH document and code context:
   - Asking to implement methods described in a paper
   - Questions comparing paper results with code output
   - Requests to replicate or code up what's in the document
   Examples: "Implement the algorithm from the paper", "How does my code compare to the paper's results?", "Code the method described in section 3"

4. GENERAL - General conversation not specifically about document or code:
   - Greetings, casual conversation
   - Meta questions about capabilities
   - General knowledge questions unrelated to research/code
   Examples: "Hello", "What can you do?", "Tell me about machine learning" (without paper context)

Respond with ONLY one word: DOCUMENT, CODE, BOTH, or GENERAL"""

    try:
        response = query_openai_api(instruction, query)
        response_clean = response.strip().upper()
        
        # Map response to enum
        if "DOCUMENT" in response_clean and "CODE" not in response_clean:
            return QueryContext.DOCUMENT
        elif "CODE" in response_clean and "DOCUMENT" not in response_clean:
            return QueryContext.CODE
        elif "BOTH" in response_clean:
            return QueryContext.BOTH
        elif "GENERAL" in response_clean:
            return QueryContext.GENERAL
        else:
            # If unclear, check for obvious keywords as fallback
            query_lower = query.lower()
            doc_keywords = ['paper', 'document', 'pdf', 'study', 'research', 'author']
            code_keywords = ['code', 'function', 'error', 'bug', 'implement', 'cell', 'notebook']
            
            has_doc = any(k in query_lower for k in doc_keywords)
            has_code = any(k in query_lower for k in code_keywords)
            
            if has_doc and has_code:
                return QueryContext.BOTH
            elif has_doc:
                return QueryContext.DOCUMENT
            elif has_code:
                return QueryContext.CODE
            else:
                return QueryContext.GENERAL
                
    except Exception as e:
        print(f"Error in query context classification: {e}")
        # Fallback to simple heuristic
        query_lower = query.lower()
        doc_keywords = ['paper', 'document', 'pdf', 'study', 'research', 'author', 'finding', 'method', 'result']
        code_keywords = ['code', 'function', 'error', 'bug', 'implement', 'cell', 'notebook', 'run', 'execute']
        
        has_doc = any(k in query_lower for k in doc_keywords)
        has_code = any(k in query_lower for k in code_keywords)
        
        if has_doc and has_code:
            return QueryContext.BOTH
        elif has_doc:
            return QueryContext.DOCUMENT
        elif has_code:
            return QueryContext.CODE
        else:
            return QueryContext.GENERAL

def is_document_query(query: str) -> bool:
    """
    Legacy function for backward compatibility.
    Returns True if query involves document context (DOCUMENT or BOTH).
    """
    context = classify_query_context(query)
    return context in [QueryContext.DOCUMENT, QueryContext.BOTH]

# Request/Response models for API endpoints
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

class ExecuteRequest(BaseModel):
    code: str
    cell_id: Optional[int] = None

class SelectNotebookRequest(BaseModel):
    filename: str

@app.get("/")
async def root():
    return {"message": "Gene Analysis IDE API"}

@app.post("/api/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """
    Upload a PDF file and create FAISS index for RAG.
    Replaces any existing PDF for this session.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Clean up any existing session data
        if session_id in session_storage:
            cleanup_session(session_id)
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        pdf_path = os.path.join(TEMP_DIR, unique_filename)
        
        # Save uploaded file
        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"PDF saved to: {pdf_path}")
        
        # Process PDF with RAG indexer
        try:
            # Extract text from PDF
            texts = extract_text_from_pdf(pdf_path)
            if not texts:
                raise ValueError("Could not extract text from PDF")
            
            # Chunk text
            chunks = chunk_text(texts)
            if not chunks:
                raise ValueError("Could not create chunks from PDF text")
            
            # Generate embeddings
            embeddings = embed_texts(chunks)
            
            # Build FAISS index
            faiss_index = build_faiss_index(embeddings)
            
            # Store in session
            session_storage[session_id] = {
                'pdf_path': pdf_path,
                'pdf_name': file.filename,
                'chunks': chunks,
                'embeddings': embeddings,
                'faiss_index': faiss_index,
                'num_pages': len(texts),
                'num_chunks': len(chunks)
            }
            
            print(f"Successfully indexed PDF for session {session_id}: {file.filename}")
            
            return {
                "status": "success",
                "message": "PDF uploaded and indexed successfully",
                "pdf_name": file.filename,
                "num_pages": len(texts),
                "num_chunks": len(chunks)
            }
            
        except Exception as e:
            # Clean up file if processing failed
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat with intelligent context handling.
    
    Supports:
    - Document context (RAG from uploaded PDFs)
    - Code context (from notebook cells)
    - Combined document + code context
    - General conversation
    """
    await websocket.accept()
    
    # Extract session_id from query params
    session_id = websocket.query_params.get("session_id", None)
    print(f"WebSocket connected with session_id: {session_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                # Parse the incoming message
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                chat_history = message_data.get("history", [])
                cell_id = message_data.get("cell_id", None)  # Current cell ID from frontend
                all_codes = message_data.get("all_codes", None)  # All edited codes from frontend
                
                if not user_message.strip():
                    await websocket.send_text("<<<ERROR>>>")
                    continue
                
                print(f"Processing message: {user_message[:100]}...")  # Truncate for logging
                print(f"Current cell: {cell_id}")
                if all_codes:
                    print(f"Received {len(all_codes)} edited code cells from frontend")
                
                # Step 0: Check if user provided explicit code context via Cmd+K
                user_provided_code, cleaned_message = extract_user_provided_code_context(user_message)
                if user_provided_code:
                    print(f"User provided explicit code context via Cmd+K (length: {len(user_provided_code)} chars)")
                    # Use the cleaned message without the embedded context
                    user_message = cleaned_message
                
                # Step 1: Classify the query context
                # Always classify properly - don't assume CODE just because there's no document
                has_document = session_id and session_id in session_storage
                query_context = classify_query_context(user_message)
                print(f"Query classified as: {query_context.value}")
                
                # Step 2: Gather appropriate context based on classification
                code_context = ""
                document_context = ""
                
                # Get code context for CODE or BOTH queries
                if query_context in [QueryContext.CODE, QueryContext.BOTH]:
                    if user_provided_code:
                        # Use ONLY the user-provided code context from Cmd+K
                        code_context = f"""## User-Selected Code Context:
```python
{user_provided_code}
```
"""
                        print(f"Using user-provided code context only (Cmd+K)")
                    else:
                        # Use current cell only (not all cells) to reduce context size
                        code_context = get_code_context(cell_id, all_codes, current_cell_only=True)
                        if code_context:
                            print(f"Retrieved code context from current cell only: {cell_id}")
                        elif not all_codes:
                            print(f"No code context available - no codes sent from UI")
                
                # Get document context for DOCUMENT or BOTH queries
                if query_context in [QueryContext.DOCUMENT, QueryContext.BOTH] and has_document:
                    document_context = get_document_context(session_id, user_message, k=5)
                    if document_context:
                        print(f"Retrieved document context from PDF")
                
                # Step 3: Build prompt with appropriate context
                messages = build_prompt(
                    user_message=user_message,
                    query_context=query_context,
                    code_context=code_context,
                    document_context=document_context,
                    chat_history=chat_history
                )
                
                # Step 4: Get streaming response from OpenAI
                response_generator = get_openai_streaming_response(messages)
                
                # Step 5: Stream each chunk to the client
                async for chunk in response_generator:
                    if chunk is not None and chunk != '':
                        await websocket.send_text(chunk)
                        # Small delay to ensure chunks are sent separately
                        await asyncio.sleep(0.02)
                
                # Send end marker
                await websocket.send_text("<<<END>>>")
                
            except json.JSONDecodeError:
                print("JSON decode error in websocket message")
                await websocket.send_text("<<<ERROR>>>")
            except Exception as e:
                print(f"Chat error: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_text("<<<ERROR>>>")
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session: {session_id}")
        # Clean up session on disconnect
        if session_id:
            cleanup_session(session_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if session_id:
            cleanup_session(session_id)

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """HTTP endpoint for streaming chat (alternative to WebSocket)"""
    try:
        messages = format_messages(request.message, [msg.dict() for msg in request.history])
        
        # Collect all chunks for HTTP response
        response_chunks = []
        async for chunk in get_openai_streaming_response(messages):
            if chunk:
                response_chunks.append(chunk)
        
        full_response = "".join(response_chunks)
        return {"response": full_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/execute")
async def execute_code(request: ExecuteRequest):
    """Execute Python code using Jupyter kernel"""
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Received execute request for cell {request.cell_id}")
        km = get_kernel_manager()
        result = km.execute_code(request.code, request.cell_id)
        logger.info(f"Execution completed, sending response back")
        return result
    except Exception as e:
        logger.error(f"Execution error: {e}")
        return {
            "outputs": [{
                "type": "error",
                "ename": "KernelError",
                "evalue": str(e),
                "traceback": [str(e)]
            }],
            "status": "error"
        }


## Streaming endpoint removed per user request

@app.post("/api/restart_kernel")
async def restart_kernel():
    """Restart the Jupyter kernel"""
    try:
        km = get_kernel_manager()
        result = km.restart_kernel()
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/kernel_status")
async def kernel_status():
    """Get kernel status"""
    try:
        km = get_kernel_manager()
        return km.get_status()
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================================
# CONTEXT RETRIEVAL HELPERS
# ============================================================================

def extract_user_provided_code_context(message: str) -> Tuple[Optional[str], str]:
    """
    Extract user-provided code context from Cmd+K inline editor.
    The inline editor formats messages as: "query\n\nContext:\ncode"
    
    Args:
        message: The user's message
    
    Returns:
        Tuple of (extracted_code_context, cleaned_message)
        If no context found, returns (None, original_message)
    """
    # Check if message contains explicit context from Cmd+K
    if "\n\nContext:\n" in message:
        parts = message.split("\n\nContext:\n", 1)
        query = parts[0].strip()
        context = parts[1].strip()
        return (context, query)
    
    return (None, message)

def get_code_context(
    current_cell_id: Optional[str] = None, 
    all_codes: Optional[Dict[str, str]] = None,
    current_cell_only: bool = True
) -> str:
    """
    Retrieve code context from notebook cells sent by the UI.
    
    Args:
        current_cell_id: The ID of the current cell (e.g., "step-1")
        all_codes: Dict of codes from UI {step_id: code}
        current_cell_only: If True, only return current cell code (default)
                          If False, return all cells (not recommended - context bloat)
    
    Returns:
        Formatted string with code context, or empty string if no codes provided
    """
    # Only use codes explicitly sent from UI - no fallback to file
    if not all_codes:
        return ""
    
    # If current_cell_only is True (default), just return the current cell
    if current_cell_only and current_cell_id and current_cell_id in all_codes:
        code = all_codes[current_cell_id]
        lines = code.splitlines()
        first_line = next((line for line in lines if line.strip()), "")
        title = first_line.strip().lstrip("#").strip() or f"Cell {current_cell_id}"
        
        return f"""## Current Cell: {title} ({current_cell_id})
```python
{code}
```
"""
    
    # Fallback: if no current cell or current_cell_only=False, return all cells
    # (This path should rarely be taken with current_cell_only=True by default)
    all_cells = []
    for step_id, code in all_codes.items():
        try:
            step_num = int(step_id.split('-')[1])
            lines = code.splitlines()
            first_line = next((line for line in lines if line.strip()), "")
            title = first_line.strip().lstrip("#").strip() or f"Cell {step_id}"
            
            all_cells.append({
                "step_id": step_id,
                "title": title,
                "code": code
            })
        except (IndexError, ValueError):
            continue
    
    all_cells.sort(key=lambda x: int(x["step_id"].split('-')[1]))
    
    if not all_cells:
        return ""
    
    # Find current cell
    current_cell = None
    other_cells = []
    
    for cell in all_cells:
        if current_cell_id and cell["step_id"] == current_cell_id:
            current_cell = cell
        else:
            other_cells.append(cell)
    
    # Build context string
    context_parts = []
    
    # Add current cell first if it exists
    if current_cell:
        context_parts.append(f"""## Current Cell: {current_cell['title']} ({current_cell['step_id']})
```python
{current_cell['code']}
```
""")
    
    # Add other cells
    if other_cells:
        context_parts.append("\n## Other Cells in Notebook:\n")
        for cell in other_cells:
            context_parts.append(f"""### {cell['title']} ({cell['step_id']})
```python
{cell['code']}
```
""")
    
    return "\n".join(context_parts)

def get_document_context(session_id: str, query: str, k: int = 5) -> str:
    """
    Retrieve relevant document chunks from the FAISS index.
    
    Args:
        session_id: The session ID with uploaded PDF
        query: The user's query
        k: Number of chunks to retrieve
    
    Returns:
        Formatted string with document context
    """
    if session_id not in session_storage:
        return ""
    
    session_data = session_storage[session_id]
    
    try:
        results = retrieve(
            session_data['faiss_index'],
            session_data['embeddings'],
            session_data['chunks'],
            query,
            k=k
        )
        
        # Format context from retrieved chunks
        context_parts = []
        for i, result in enumerate(results, 1):
            page = result['meta']['page']
            text = result['text']
            score = result['score']
            context_parts.append(
                f"[Excerpt {i} - Page {page}, Relevance: {score:.2f}]\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    except Exception as e:
        print(f"Error retrieving document context: {e}")
        return ""

def build_prompt(
    user_message: str,
    query_context: QueryContext,
    code_context: str = "",
    document_context: str = "",
    chat_history: List[Dict] = None
) -> List[Dict[str, str]]:
    """
    Build the prompt based on query context type.
    
    Args:
        user_message: The user's query
        query_context: The classified context type
        code_context: Code context from notebook cells
        document_context: Document context from RAG
        chat_history: Previous chat messages
    
    Returns:
        Formatted messages for the LLM
    """
    
    if query_context == QueryContext.BOTH:
        # Query involves both document and code
        # Check if we have the necessary contexts
        if not document_context and not code_context:
            # Neither context available - inform user
            no_context_message = f"""The user is asking about both a document and code, but:
- No document has been uploaded yet
- No code is available in the notebook

User Question: {user_message}

Please inform the user that they need to:
1. Upload a PDF document using the upload button
2. Have code in the notebook cells
Then you'll be able to help them connect the document with the implementation."""
            return format_messages(no_context_message, chat_history)
        elif not document_context:
            # Only code context available - treat as code query
            augmented_message = f"""The user is asking about code implementation.

### Code Context (from notebook cells):
{code_context}

### User Question:
{user_message}

Please provide a helpful answer about the code. Include:
1. Clear explanations of how the code works
2. Code examples or modifications if needed
3. Best practices and suggestions
4. Debugging help if the question involves an error
Note: The user mentioned a document, but none has been uploaded yet."""
            return format_messages(augmented_message, chat_history)
        elif not code_context:
            # Only document context available - treat as document query
            augmented_message = f"""Based on the following excerpts from the uploaded PDF document, please answer the user's question.

### Document Context:
{document_context}

### User Question:
{user_message}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information, please indicate that.
Note: The user mentioned code, but no code cells are available in the notebook yet."""
            return format_messages(augmented_message, chat_history)
        else:
            # Both contexts available - proceed normally
            augmented_message = f"""The user has asked a question that involves BOTH the research paper/document AND code implementation.

### Document Context (from uploaded PDF):
{document_context}

### Code Context (from notebook cells):
{code_context}

### User Question:
{user_message}

Please provide a comprehensive answer that:
1. References relevant information from the document context
2. Addresses the code/implementation aspects
3. Provides code examples or modifications if needed
4. Connects the paper's concepts to practical implementation"""
            
            return format_messages(augmented_message, chat_history)
    
    elif query_context == QueryContext.DOCUMENT:
        # Document-only query
        if not document_context:
            # User is asking about a document but none is uploaded
            no_doc_message = f"""The user is asking about a document/paper, but no document has been uploaded yet.

User Question: {user_message}

Please politely inform the user that they need to upload a PDF document first using the upload button in the chat interface, and then you'll be able to answer their questions about it."""
            return format_messages(no_doc_message, chat_history)
        
        augmented_message = f"""Based on the following excerpts from the uploaded PDF document, please answer the user's question.

### Document Context:
{document_context}

### User Question:
{user_message}

"""
        if is_document_summary_query(user_message):
            augmented_message += PDF_SUMMARY_FORMAT
        
        augmented_message += """Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information, please indicate that."""
        
        return format_messages(augmented_message, chat_history)
    
    elif query_context == QueryContext.CODE:
        # Code-only query
        if not code_context:
            # User is asking about code but no code is available
            no_code_message = f"""The user is asking about code, but no code cells are currently available in the notebook.

User Question: {user_message}

Please provide a helpful general answer about the coding question. You can:
1. Explain concepts and approaches
2. Provide example code snippets
3. Offer best practices
But note that you don't have access to their specific notebook code at the moment."""
            return format_messages(no_code_message, chat_history)
        
        augmented_message = f"""The user is asking about code implementation.

### Code Context (from notebook cells):
{code_context}

### User Question:
{user_message}

Please provide a helpful answer about the code. Include:
1. Clear explanations of how the code works
2. Code examples or modifications if needed
3. Best practices and suggestions
4. Debugging help if the question involves an error"""
        
        return format_messages(augmented_message, chat_history)
    
    else:
        # General query - no special context needed
        return format_messages(user_message, chat_history)

# -----------------------------
# Notebook helper endpoints
# -----------------------------

def _resolve_notebook_path(path: Optional[str] = None) -> str:
    """Resolve the absolute path to the notebook file.

    Uses the currently selected notebook from NOTEBOOKS_DIR if no path provided.
    """
    global CURRENT_NOTEBOOK
    if path and path.strip():
        return os.path.abspath(path)
    # Use current selected notebook from notebooks directory
    return os.path.join(NOTEBOOKS_DIR, CURRENT_NOTEBOOK)


@app.get("/api/notebook/cells")
async def list_notebook_cells(path: Optional[str] = None):
    """List code cells from a Jupyter notebook as step candidates."""
    nb_path = _resolve_notebook_path(path)
    if not os.path.exists(nb_path):
        raise HTTPException(status_code=404, detail=f"Notebook not found: {nb_path}")

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read notebook: {e}")

    cells = nb_data.get("cells", [])
    result = []
    step_index = 0
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            first_line = next((line for line in source if str(line).strip() != ""), "")
        else:
            # string
            lines = str(source).splitlines()
            first_line = next((line for line in lines if line.strip() != ""), "")

        title = first_line.strip()
        # Strip leading comment markers for a cleaner title
        if title.startswith("#"):
            title = title.lstrip("#").strip()

        step_index += 1
        result.append({
            "index": idx,               # actual notebook cell index
            "stepNumber": step_index,   # 1-based step sequence among code cells
            "title": title or f"Cell {idx}",
            "description": title or "Notebook code cell",
        })

    return {"notebook": nb_path, "steps": result}


@app.get("/api/notebook/cell/{index}")
async def get_notebook_cell(index: int, path: Optional[str] = None):
    """Fetch the full source of a specific notebook cell by its absolute cell index."""
    nb_path = _resolve_notebook_path(path)
    if not os.path.exists(nb_path):
        raise HTTPException(status_code=404, detail=f"Notebook not found: {nb_path}")

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read notebook: {e}")

    cells = nb_data.get("cells", [])
    if index < 0 or index >= len(cells):
        raise HTTPException(status_code=404, detail=f"Cell index out of range: {index}")

    cell = cells[index]
    if cell.get("cell_type") != "code":
        raise HTTPException(status_code=400, detail=f"Cell {index} is not a code cell")

    source = cell.get("source", [])
    if isinstance(source, list):
        code = "".join(source)
    else:
        code = str(source)

    return {"index": index, "source": code}


# -----------------------------
# Notebook management endpoints
# -----------------------------

@app.post("/api/notebooks/upload")
async def upload_notebook(file: UploadFile = File(...)):
    """Upload a new Jupyter notebook file"""
    try:
        content = await file.read()

        if USE_AZURE_STORAGE:
            # Upload to Azure Blob Storage
            result = blob_manager.upload_notebook(file.filename, content)

            # Also save to local cache for Jupyter kernel
            os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
            file_path = os.path.join(NOTEBOOKS_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(content)

            return result
        else:
            # Fallback: local storage only
            # Validate file extension
            if not file.filename.endswith('.ipynb'):
                raise HTTPException(status_code=400, detail="Only .ipynb files are allowed")

            # Validate file content is valid JSON
            try:
                nb_data = json.loads(content)
                if "cells" not in nb_data or "metadata" not in nb_data:
                    raise HTTPException(status_code=400, detail="Invalid Jupyter notebook format")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="File is not valid JSON")

            # Ensure notebooks directory exists
            os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

            # Save the file locally
            file_path = os.path.join(NOTEBOOKS_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(content)

            # Get file info
            file_stat = os.stat(file_path)

            return {
                "status": "success",
                "filename": file.filename,
                "size": file_stat.st_size,
                "uploaded_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/notebooks")
async def list_notebooks():
    """List all available notebooks"""
    try:
        if USE_AZURE_STORAGE:
            # Get list from Azure Blob Storage
            notebooks = blob_manager.list_notebooks()

            # Add is_current flag
            for nb in notebooks:
                nb["is_current"] = nb["filename"] == CURRENT_NOTEBOOK

            return {"notebooks": notebooks, "current": CURRENT_NOTEBOOK}
        else:
            # Fallback: local storage
            if not os.path.exists(NOTEBOOKS_DIR):
                return {"notebooks": [], "current": CURRENT_NOTEBOOK}

            notebooks = []
            for filename in os.listdir(NOTEBOOKS_DIR):
                if filename.endswith('.ipynb'):
                    file_path = os.path.join(NOTEBOOKS_DIR, filename)
                    file_stat = os.stat(file_path)
                    notebooks.append({
                        "filename": filename,
                        "size": file_stat.st_size,
                        "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        "is_current": filename == CURRENT_NOTEBOOK
                    })

            # Sort by modified time, most recent first
            notebooks.sort(key=lambda x: x["modified_at"], reverse=True)

            return {"notebooks": notebooks, "current": CURRENT_NOTEBOOK}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list notebooks: {str(e)}")


@app.post("/api/notebooks/select")
async def select_notebook(request: SelectNotebookRequest):
    """Select a notebook as the current active notebook"""
    global CURRENT_NOTEBOOK

    filename = request.filename

    try:
        if USE_AZURE_STORAGE:
            # Download from Azure Blob Storage to local cache
            try:
                content = blob_manager.download_notebook(filename)

                # Ensure local cache directory exists
                os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

                # Save to local cache for Jupyter kernel
                file_path = os.path.join(NOTEBOOKS_DIR, filename)
                with open(file_path, "wb") as f:
                    f.write(content)

            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Notebook not found: {filename}")

        else:
            # Fallback: local storage
            file_path = os.path.join(NOTEBOOKS_DIR, filename)
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"Notebook not found: {filename}")

        # Set as current notebook
        CURRENT_NOTEBOOK = filename

        # Load and return basic info about the selected notebook
        file_path = os.path.join(NOTEBOOKS_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        cell_count = len(nb_data.get("cells", []))
        code_cell_count = sum(1 for c in nb_data.get("cells", []) if c.get("cell_type") == "code")

        return {
            "status": "success",
            "current": CURRENT_NOTEBOOK,
            "cell_count": cell_count,
            "code_cell_count": code_cell_count
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to select notebook: {str(e)}")


@app.delete("/api/notebooks/{filename}")
async def delete_notebook(filename: str):
    """Delete a notebook file"""
    global CURRENT_NOTEBOOK

    # Prevent deletion of current notebook
    if filename == CURRENT_NOTEBOOK:
        raise HTTPException(status_code=400, detail="Cannot delete the currently active notebook")

    try:
        if USE_AZURE_STORAGE:
            # Delete from Azure Blob Storage
            result = blob_manager.delete_notebook(filename)

            # Also delete from local cache if exists
            file_path = os.path.join(NOTEBOOKS_DIR, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

            return result
        else:
            # Fallback: local storage
            file_path = os.path.join(NOTEBOOKS_DIR, filename)
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"Notebook not found: {filename}")

            os.remove(file_path)
            return {"status": "success", "message": f"Deleted {filename}"}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Notebook not found: {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete notebook: {str(e)}")


@app.get("/api/notebooks/current")
async def get_current_notebook():
    """Get information about the current notebook"""
    try:
        if USE_AZURE_STORAGE:
            # Get metadata from Azure Blob Storage
            try:
                metadata = blob_manager.get_notebook_metadata(CURRENT_NOTEBOOK)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Current notebook not found")

            # Read from local cache to get cell counts
            file_path = os.path.join(NOTEBOOKS_DIR, CURRENT_NOTEBOOK)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    nb_data = json.load(f)
                cell_count = len(nb_data.get("cells", []))
                code_cell_count = sum(1 for c in nb_data.get("cells", []) if c.get("cell_type") == "code")
            else:
                # If not in cache, download and read
                content = blob_manager.download_notebook(CURRENT_NOTEBOOK)
                nb_data = json.loads(content.decode('utf-8'))
                cell_count = len(nb_data.get("cells", []))
                code_cell_count = sum(1 for c in nb_data.get("cells", []) if c.get("cell_type") == "code")

            return {
                "filename": metadata["filename"],
                "size": metadata["size"],
                "modified_at": metadata["modified_at"],
                "cell_count": cell_count,
                "code_cell_count": code_cell_count
            }
        else:
            # Fallback: local storage
            file_path = _resolve_notebook_path()

            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Current notebook not found")

            file_stat = os.stat(file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                nb_data = json.load(f)

            cell_count = len(nb_data.get("cells", []))
            code_cell_count = sum(1 for c in nb_data.get("cells", []) if c.get("cell_type") == "code")

            return {
                "filename": CURRENT_NOTEBOOK,
                "size": file_stat.st_size,
                "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "cell_count": cell_count,
                "code_cell_count": code_cell_count
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read notebook: {str(e)}")


# -----------------------------
# Workspace file management endpoints
# -----------------------------

@app.post("/api/workspace/upload")
async def upload_workspace_file(file: UploadFile = File(...)):
    """Upload any file to workspace (CSV, JSON, TXT, PY, etc.)"""
    try:
        content = await file.read()

        if USE_AZURE_STORAGE:
            # Upload to Azure uploads container
            result = blob_manager.upload_file(file.filename, content)

            # Also save to local workspace for Jupyter kernel access
            os.makedirs(WORKSPACE_DIR, exist_ok=True)
            file_path = os.path.join(WORKSPACE_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(content)

            return result
        else:
            # Fallback: local storage only
            os.makedirs(WORKSPACE_DIR, exist_ok=True)
            file_path = os.path.join(WORKSPACE_DIR, file.filename)

            with open(file_path, "wb") as f:
                f.write(content)

            file_stat = os.stat(file_path)
            return {
                "status": "success",
                "filename": file.filename,
                "size": file_stat.st_size,
                "uploaded_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/workspace/files")
async def list_workspace_files():
    """List all workspace files"""
    try:
        if USE_AZURE_STORAGE:
            # Get list from Azure uploads container
            files = blob_manager.list_files()
            return {"files": files}
        else:
            # Fallback: local storage
            if not os.path.exists(WORKSPACE_DIR):
                return {"files": []}

            files = []
            for filename in os.listdir(WORKSPACE_DIR):
                file_path = os.path.join(WORKSPACE_DIR, filename)
                if os.path.isfile(file_path):
                    file_stat = os.stat(file_path)
                    files.append({
                        "filename": filename,
                        "size": file_stat.st_size,
                        "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })

            # Sort by modified time, most recent first
            files.sort(key=lambda x: x["modified_at"], reverse=True)
            return {"files": files}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@app.delete("/api/workspace/{filename}")
async def delete_workspace_file(filename: str):
    """Delete a workspace file"""
    try:
        if USE_AZURE_STORAGE:
            result = blob_manager.delete_file(filename)

            # Also delete from local cache
            file_path = os.path.join(WORKSPACE_DIR, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

            return result
        else:
            file_path = os.path.join(WORKSPACE_DIR, filename)
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {filename}")

            os.remove(file_path)
            return {"status": "success", "message": f"Deleted {filename}"}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Increase timeout for long-running code execution in low-memory environments
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300,  # 5 minutes keep-alive timeout
        workers=1  # Use single worker to save memory
    )