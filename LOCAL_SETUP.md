# Local Development Setup Guide

Complete documentation of all steps taken to set up and run the Alzheimer's Analysis Pipeline locally.

---

## Prerequisites Check

Before starting, ensure you have:

- **Python 3.8+** installed
  ```bash
  python3 --version
  ```
  
- **Node.js 18+** installed
  ```bash
  node --version
  npm --version
  ```
  
- **OpenAI API Key** from https://platform.openai.com/api-keys

---

## Step-by-Step Setup Process

### Step 1: Install Python Backend Dependencies

**Location:** Project root directory

**Command:**
```bash
cd backend
pip install -r ../requirements.txt
```

**Why?** The root `requirements.txt` contains ALL necessary dependencies including:
- FastAPI web framework
- Jupyter kernel management libraries (jupyter_client, ipykernel, traitlets, etc.)
- Data science libraries (numpy, pandas, matplotlib, seaborn)
- OpenAI API client
- WebSocket support
- All other runtime dependencies

**Note:** Do NOT use `backend/requirements.txt` as it's incomplete and missing critical kernel management libraries.

**Expected output:**
- Packages download and install
- May show warnings about dependencies, but should complete successfully

---

### Step 2: Install Additional Backend Dependency

**Why?** The root `requirements.txt` is missing `python-multipart` which is needed for FastAPI file uploads.

**Command:**
```bash
pip install python-multipart==0.0.6
```

**Expected output:**
```
Successfully installed python-multipart-0.0.6
```

---

### Step 3: Create Environment File with OpenAI API Key

**Location:** `backend/` directory

**Command:**
```bash
cd backend
echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
```

**Important:** 
- Replace `your_actual_api_key_here` with your actual API key
- The `.env` file is gitignored and will NOT be committed to the repository
- This file is only for local development

**Verify creation:**
```bash
cat .env
```

---

### Step 4: Fix Backend Code to Load .env File

**Why?** The original `backend/main.py` doesn't automatically load the `.env` file, causing the API key to not be found.

**File to modify:** `backend/main.py`

**What to add:** Insert this code at the VERY TOP of the file (before any imports):

```python
# Load environment variables from .env file FIRST, before any imports
import os

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

```

**Why this works:**
- Loads environment variables BEFORE importing modules that need them
- Uses `python-dotenv` if available, otherwise manually parses the file
- Safe for production deployments (Azure uses environment variables directly)

---

### Step 5: Install Node.js (if not already installed)

**Why?** Frontend requires Node.js and npm to install dependencies and run the dev server.

**Method 1: Official Download (Recommended)**
1. Visit https://nodejs.org/
2. Download the LTS (Long Term Support) version
3. Run the installer
4. Follow installation prompts

**Method 2: Homebrew (if you have brew)**
```bash
brew install node
```

**Verify installation:**
```bash
node --version  # Should show v18.0.0 or higher
npm --version   # Should show 9.0.0 or higher
```

---

### Step 6: Install Frontend Dependencies

**Location:** Project root directory

**Command:**
```bash
npm install
```

**What this does:**
- Reads `package.json`
- Downloads and installs all dependencies including:
  - React and React DOM
  - TypeScript
  - Vite (build tool)
  - Tailwind CSS
  - UI libraries (lucide-react, recharts, axios)
  - All dev dependencies

**Expected output:**
```
added 282 packages, and audited 283 packages in 2s
```

**Note:** You may see vulnerability warnings. These are typically dev dependencies and won't affect functionality.

---

### Step 7: Start Backend Server

**Location:** `backend/` directory

**Command:**
```bash
cd backend
python3 main.py
```

**Expected output:**
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Important:**
- Keep this terminal window open
- Backend runs on **http://localhost:8000**
- Jupyter kernel initializes automatically

---

### Step 8: Start Frontend Development Server

**Location:** Project root directory

**Command (in a NEW terminal window):**
```bash
npm run dev
```

**Expected output:**
```
  VITE v5.0.0  ready in xxx ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

**Important:**
- Keep this terminal window open
- Frontend runs on **http://localhost:3000**
- Auto-reloads when you change code

---

### Step 9: Access the Application

**Open in Browser:**
1. Open any web browser (Chrome, Firefox, Safari, Edge)
2. Navigate to: **http://localhost:3000**
3. Wait for the page to load

**What you should see:**
- Three-panel interface:
  - **Left Panel:** Alzheimer's Analysis Pipeline steps
  - **Center Panel:** Code editor with notebook cells
  - **Right Panel:** AI Assistant chat
- "Connected" status indicator in chat panel
- AI greeting message: "Hello! I'm your AI assistant for Alzheimer's disease research..."

---

## Verification Tests

### Test 1: Backend API Endpoint

**Command:**
```bash
curl http://localhost:8000/
```

**Expected response:**
```json
{"message":"Alzheimer's Analysis Pipeline API"}
```

### Test 2: Kernel Status

**Command:**
```bash
curl http://localhost:8000/api/kernel_status
```

**Expected response:**
```json
{"status": "running", "kernel_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}
```

### Test 3: Notebook Cells Endpoint

**Command:**
```bash
curl http://localhost:8000/api/notebook/cells
```

**Expected response:**
```json
{
    "notebook": "/path/to/colab.ipynb",
    "steps": [
        {"index": 1, "stepNumber": 1, "title": "...", ...},
        ...
    ]
}
```

### Test 4: Frontend Accessibility

**Action:** Open browser developer tools (F12 or Right-click → Inspect)

**What to verify:**
- No console errors
- Network tab shows successful API calls
- Three panels render correctly
- Chat WebSocket connection established

---

## Troubleshooting Common Issues

### Issue: Backend Won't Start

**Symptom:**
```
ValueError: OPENAI_API_KEY environment variable is required
```

**Solutions:**
1. Ensure `.env` file exists: `ls -la backend/.env`
2. Check API key is correct: `cat backend/.env`
3. Verify `main.py` has the `.env` loading code at the top
4. Try: `python3 main.py` instead of `python main.py`

---

### Issue: Port Already in Use

**Symptom:**
```
ERROR: [Errno 48] Address already in use
```

**Solutions:**

**Backend:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or change port in backend/main.py line 264
```

**Frontend:**
```bash
# Find process using port 3000
lsof -i :3000

# Kill it
kill -9 <PID>

# Or change port in vite.config.ts line 8
```

---

### Issue: Frontend Can't Connect to Backend

**Symptom:**
- Chat shows "Disconnected"
- API calls fail
- Console shows network errors

**Solutions:**
1. Verify backend is running: `curl http://localhost:8000/`
2. Check vite.config.ts has correct proxy settings
3. Ensure both servers are running simultaneously
4. Check for CORS errors in browser console

---

### Issue: Jupyter Kernel Errors

**Symptom:**
- Code execution fails
- Kernel status shows error

**Solutions:**
1. Restart backend server
2. Check python dependencies are installed
3. Verify ipykernel and jupyter_client versions
4. Try: `pip install --upgrade jupyter_client ipykernel`

---

### Issue: Missing Python Dependencies

**Symptom:**
```
ModuleNotFoundError: No module named 'xxx'
```

**Solutions:**
1. Reinstall from root requirements.txt
2. Check you used root file, not backend/requirements.txt
3. Try: `pip install -r requirements.txt --force-reinstall`

---

## Development Workflow

### Making Changes

**Frontend changes:**
- Edit files in `src/`
- Changes auto-reload in browser
- Check browser console for errors

**Backend changes:**
- Edit files in `backend/`
- Restart backend server to apply changes
- Test with `curl` or browser

### Stopping the Application

**Backend:**
- Go to backend terminal
- Press `Ctrl+C`

**Frontend:**
- Go to frontend terminal
- Press `Ctrl+C`

---

## File Structure Reference

```
CMU-SEM-49772-alzheimers-pipeline/
├── backend/
│   ├── .env                          # YOUR API KEY (gitignored)
│   ├── main.py                       # Backend server (MODIFIED to load .env)
│   ├── kernel_manager.py            # Jupyter kernel management
│   ├── models/
│   │   └── openai_chat.py          # OpenAI integration
│   └── requirements.txt             # DON'T USE THIS ONE
├── src/                              # Frontend React app
│   ├── components/                   # UI components
│   ├── App.tsx                       # Main app
│   └── ...
├── requirements.txt                  # USE THIS ONE for Python deps
├── package.json                      # Frontend dependencies
├── vite.config.ts                    # Vite configuration
├── colab.ipynb                       # Analysis pipeline notebook
├── SETUP_GUIDE.md                    # Quick reference
└── LOCAL_SETUP.md                    # This file
```

---

## Important Files Modified

### backend/main.py
**Lines 1-16 added:**
- Code to load `.env` file before imports
- Manual parsing fallback if python-dotenv not installed
- Environment variable loading logic

**Why:** Makes local development easier without affecting production deployments.

---

## Summary of All Steps Executed

1. ✅ Installed Python dependencies from root `requirements.txt`
2. ✅ Installed additional `python-multipart` package
3. ✅ Created `backend/.env` file with OpenAI API key
4. ✅ Modified `backend/main.py` to load `.env` file
5. ✅ Installed Node.js 24.11.0
6. ✅ Installed frontend dependencies with `npm install`
7. ✅ Started backend server on port 8000
8. ✅ Started frontend dev server on port 3000
9. ✅ Verified all endpoints working
10. ✅ Tested application in browser
11. ✅ Confirmed WebSocket connection
12. ✅ Verified Jupyter kernel running

---

## Next Steps

After setup is complete:

1. **Explore the interface:**
   - Click through pipeline steps
   - Try running code cells
   - Chat with AI assistant

2. **Read the documentation:**
   - `Readme.md` for architecture overview
   - `SETUP_GUIDE.md` for quick reference

3. **Customize:**
   - Modify notebook cells in `colab.ipynb`
   - Adjust UI in `src/components/`
   - Add features to backend in `backend/`

---

## Support

If you encounter issues not covered here:

1. Check browser console for errors (F12)
2. Check backend terminal for error messages
3. Check frontend terminal for build errors
4. Verify all prerequisites are installed correctly
5. Review this entire document again

---

**Last Updated:** January 2025
**Setup Time:** ~15-20 minutes
**Tested On:** macOS with Python 3.9 and Node.js 24

