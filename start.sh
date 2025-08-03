#!/bin/bash

# Load environment variables (Render sets them automatically, so this is optional on Render)
# source .env

# Start the FastAPI app
uv run uvicorn main:app --host 0.0.0.0 --port 10000
