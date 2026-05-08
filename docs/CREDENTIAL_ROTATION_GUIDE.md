# Credential Rotation Guide

**Date:** 2026-04-15
**Status:** Credentials in git repo are SAFE (.env is in .gitignore)
**Action needed:** The local PROD_SETUP working copy had .env with live secrets — delete that copy or purge credentials from it.

## Credentials to Rotate (if compromised)

| Credential | Where to Generate | VMs to Update |
|-----------|------------------|---------------|
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys | Both VMs |
| `MONGODB_URI` | MongoDB Atlas → Database Access → Edit Password | Both VMs |
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | AWS IAM → Users → Security Credentials → Create Access Key | Both VMs |

## Rotation Steps

1. **Generate new credential** from the provider
2. **Update sandbox first:**
   ```bash
   ssh -i ai_assistant_sandbox.pem ubuntu@54.197.189.113
   nano /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent/.env
   # Update the key, save
   # Restart: kill $(lsof -t -i:8001) && cd /home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent && nohup uvicorn gateway.app:app --host 0.0.0.0 --port 8001 --workers 1 &
   curl http://localhost:8001/health  # verify
   ```
3. **Test on sandbox** — run a query to confirm
4. **Update prod:**
   ```bash
   ssh -i ai_assistant.pem ubuntu@13.217.22.125
   nano /home/ubuntu/vcsai/unified-rag-agent/.env
   sudo systemctl restart rag-agent.service
   curl http://localhost:8001/health
   ```
5. **Revoke old credential** from the provider
6. **Delete local PROD_SETUP .env** — `rm C:\Users\...\PROD_SETUP\unified-rag-agent\.env`

## Current Status
- GitHub repo: CLEAN (`.env` in `.gitignore`, never committed)
- Prod VM: credentials in `.env` (server-only, not in git)
- Sandbox VM: credentials in `.env` (server-only, not in git)
- Local PROD_SETUP copy: HAS `.env` — **delete this file**
