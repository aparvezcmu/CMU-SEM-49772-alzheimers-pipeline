# Load environment variables from .env file FIRST, before any imports
import os

def _load_env_fallback():
    """Fallback loader for .env that gives a clearer error if encoding is bad."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    try:
        # Try the common UTF‑8 encoding first (with BOM support)
        with open(env_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    except UnicodeDecodeError:
        # This usually means the file was saved as UTF‑16 or another non‑UTF‑8 encoding.
        # We surface a clear message instead of a cryptic stack trace.
        raise RuntimeError(
            "Failed to read '.env' file due to encoding issues. "
            "Please re-save the file as UTF-8 (without BOM) and try again."
        )

try:
    from dotenv import load_dotenv
    try:
        # Explicitly tell python-dotenv to expect UTF‑8, and fall back if that fails.
        load_dotenv(encoding="utf-8")
    except UnicodeDecodeError:
        _load_env_fallback()
except ImportError:
    # If python-dotenv is not installed, manually load .env file
    _load_env_fallback()


from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from typing import List, Optional
import os
from datetime import datetime
from pathlib import Path
from models.openai_chat import (
    get_openai_streaming_response,
    format_messages,
    summarize_plot_with_image,
)
from kernel_manager import get_kernel_manager
import base64

app = FastAPI(title="Alzheimer's Analysis Pipeline API")
BASE_DIR = Path(__file__).resolve().parent.parent
PLOT_UPLOAD_DIR = BASE_DIR / "uploads" / "plots"
PLOT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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

@app.get("/")
async def root():
    return {"message": "Alzheimer's Analysis Pipeline API"}


@app.post("/api/plots/upload")
async def upload_plot(file: UploadFile = File(...)):
    """Upload a PNG plot for backend processing/storage."""
    if file.content_type not in ("image/png", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PNG files are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    safe_name = file.filename or "plot.png"
    filename = f"{timestamp}_{safe_name}".replace(" ", "_")
    file_path = PLOT_UPLOAD_DIR / filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save plot: {exc}")

    base64_image = base64.b64encode(contents).decode("utf-8")
    summary = None
    try:
        summary_prompt = "Summarize the key trends and implications shown in this graph."
        summary = await summarize_plot_with_image(base64_image, summary_prompt)
    except Exception as exc:
        print(f"Plot summary generation failed: {exc}")

    return {
        "status": "success",
        "filename": filename,
        "originalName": file.filename,
        "size": len(contents),
        "storedPath": str(file_path),
        "base64": base64_image,
        "summary": summary,
    }

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

def _resolve_notebook_path(path: Optional[str]) -> str:
    """Resolve the absolute path to the notebook file.

    Defaults to the project's root-level 'colab.ipynb' if no path provided.
    This backend typically runs from the 'backend' directory, so we go one level up.
    """
    if path and path.strip():
        return os.path.abspath(path)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.abspath(os.path.join(backend_dir, "..", "colab.ipynb"))
    return default_path


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