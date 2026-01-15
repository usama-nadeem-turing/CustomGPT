# OpenAI API Setup Guide

## Getting Your OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign in or create an account
3. Navigate to "API Keys" in the left sidebar
4. Click "Create new secret key"
5. Copy the generated key (it starts with `sk-`)

## Setting Up Your API Key

### Option 1: Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="sk-your-api-key-here"
```

**Windows Command Prompt:**
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
```

**Permanent Setup (Windows):**
1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Click "Environment Variables"
3. Under "User variables", click "New"
4. Variable name: `OPENAI_API_KEY`
5. Variable value: `sk-your-api-key-here`

### Option 2: .env File (Most Secure)

1. Create a file named `.env` in your project directory
2. Add this line to the file:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```
3. Make sure `.env` is in your `.gitignore` file to keep it secure

### Option 3: Direct in Code (Not Recommended)

```python
response = call_openai_api("Your prompt", api_key="sk-your-api-key-here")
```

## Installation

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your API key using one of the methods above

3. Run the script:
   ```bash
   python main.py
   ```

## Security Notes

- Never commit your API key to version control
- The `.env` file is automatically ignored by git
- Keep your API key private and secure 