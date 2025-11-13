# Quick Setup Guide

This document provides a concise checklist for setting up the Alzheimer's Analysis Pipeline locally.

## ✅ Setup Checklist

### Pre-Installation Requirements
- [ ] Python 3.8+ installed
- [ ] Node.js 18+ installed
- [ ] OpenAI API key obtained from https://platform.openai.com/api-keys

### Installation Steps

1. **Clone/Download the repository**
   ```bash
   cd CMU-SEM-49772-alzheimers-pipeline
   ```

2. **Install Python dependencies**
   ```bash
   cd backend
   pip install -r ../requirements.txt
   ```

3. **Create `.env` file**
   ```bash
   echo "OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_KEY_HERE" > .env
   ```

4. **Install Node.js dependencies**
   ```bash
   cd .. # back to project root
   npm install
   ```

### Running the Application

**Terminal 1 - Backend:**
```bash
cd backend
python3 main.py
```
✅ Backend running on: http://localhost:8000

**Terminal 2 - Frontend:**
```bash
npm run dev
```
✅ Frontend running on: http://localhost:3000

**Browser:**
- Open http://localhost:3000
- Verify all three panels load correctly

## 🧪 Quick Verification

```bash
# Test backend API
curl http://localhost:8000/
# Expected: {"message":"Alzheimer's Analysis Pipeline API"}

# Test kernel status
curl http://localhost:8000/api/kernel_status
# Expected: {"status": "running", "kernel_id": "..."}
```

## 📝 Important Notes

- **Use root `requirements.txt`** (not `backend/requirements.txt`) for complete setup
- **`.env` file** must be in `backend/` directory
- **Two terminals needed**: one for backend, one for frontend
- **Code modification**: `backend/main.py` loads `.env` file automatically for local dev

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend won't start | Check `.env` file exists with valid API key |
| Frontend won't start | Run `npm install` again |
| Port already in use | Kill process or change port in config |
| Dependencies missing | Use root `requirements.txt`, not `backend/requirements.txt` |

## 📚 Additional Documentation

- Full setup instructions: [Readme.md](Readme.md)
- Project structure: [Readme.md#architecture](Readme.md#architecture)
- API endpoints: [Readme.md#api-endpoints](Readme.md#api-endpoints)

