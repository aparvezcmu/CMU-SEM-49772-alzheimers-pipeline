# Load environment variables from .env file FIRST, before any imports
import os
import shutil
from datetime import datetime
import pickle
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from typing import List, Optional

from models.openai_chat import get_openai_streaming_response, format_messages
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
    return {"message": "Alzheimer's Analysis Pipeline API"}

@app.get("/api/checkpdf")
async def check_pdf(query: str = ""):
    # Sample API call: http://localhost:8000/api/checkpdf?query=What is the main objective of the paper?
    #pdf_path = "/Users/tanya/f1.pdf"  # replace with a test paper
    #texts = extract_text_from_pdf(pdf_path)
    #print(texts)
    #chunks = chunk_text(texts)
    # write chunks to a file pickle format
    #with open("/Users/tanya/chunks.pkl", "wb") as f:
    #    pickle.dump(chunks, f)
    #embs = embed_texts(chunks)
    # write embs to a file pickle format
    #with open("/Users/tanya/embs.pkl", "wb") as f:
    #    pickle.dump(embs, f)

    with open("/Users/tanya/embs.pkl", "rb") as f:
        embs = pickle.load(f)
    with open("/Users/tanya/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    #index = build_faiss_index(embs)
    # write index to a file pickle format
    #with open("/Users/tanya/index.pkl", "wb") as f:
    #    pickle.dump(index, f)

    with open("/Users/tanya/index.pkl", "rb") as f:
        index = pickle.load(f)
    results = retrieve(index, embs, chunks, query, k=10)

    response = ""
    for r in results:
        response += f"[Page {r['meta']['page']}] (score={r['score']:.2f}) {r['text'][:200]}...;\n"
    print(response)
    return {"status": "success", "response": response}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                # Parse the incoming message
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                chat_history = message_data.get("history", [])
                
                if not user_message.strip():
                    await websocket.send_text("<<<ERROR>>>")
                    continue
                
                # Format messages for OpenAI
                messages = format_messages(user_message, chat_history)
                
                # Get streaming response from OpenAI
                print(f"Processing message: {user_message}")
                response_generator = get_openai_streaming_response(messages)
                
                # Stream each chunk to the client with proper async handling
                async for chunk in response_generator:
                    if chunk is not None and chunk != '':
                        print(f"Sending chunk: '{chunk}'")
                        await websocket.send_text(chunk)
                        # Small delay to ensure chunks are sent separately
                        await asyncio.sleep(0.02)
                
                # Send end marker
                await websocket.send_text("<<<END>>>")
                
            except json.JSONDecodeError:
                await websocket.send_text("<<<ERROR>>>")
            except Exception as e:
                print(f"Chat error: {e}")
                await websocket.send_text("<<<ERROR>>>")
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

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