# 🔧 How to Fix the API Quota Error (429)

## Problem
You're seeing this error:
```
API Error: 429 - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota..."
  }
}
```

## Root Cause
Your Google Gemini API key has exceeded its monthly quota. This could be because:
1. ❌ **Hardcoded API key was exposed** (it was visible in the source code)
2. ❌ Too many test requests were made
3. ❌ The free tier quota was exhausted

## Solution

### Step 1: Get a New API Key
1. Go to [Google AI Studio](https://ai.google.dev/)
2. Click "Get API Key"
3. Create a new project or select existing one
4. Generate a new API key
5. Copy it (keep it secure!)

### Step 2: Set Environment Variable

**PowerShell (Windows):**
```powershell
$env:GENAI_API_KEY='your-new-api-key-here'
```

**Command Prompt (CMD):**
```cmd
set GENAI_API_KEY=your-new-api-key-here
```

**Linux/Mac (Bash):**
```bash
export GENAI_API_KEY='your-new-api-key-here'
```

### Step 3: Restart the Backend
```bash
cd mindlink_backend
python -m uvicorn main:app --reload
```

### Step 4: Test It
Try sending a message in the chat - it should work now!

## What We Fixed
✅ Removed hardcoded API key from source code
✅ Added rate limiting (3-second cooldown between messages)
✅ Added better error messages for quota exhaustion
✅ Added environment variable support for secure API key management

## Important!
- **NEVER commit API keys to Git**
- **NEVER share your API key publicly**
- Always use environment variables for secrets
- Check your [API usage](https://ai.dev/rate-limit) regularly
