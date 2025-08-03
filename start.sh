#!/bin/bash

# Load environment variables (Render sets them automatically, so this is optional on Render)
# source .env

# Start the FastAPI app
#!/bin/bash
exec gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT

