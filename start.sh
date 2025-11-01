#!/usr/bin/env bash
# Use Gunicorn with Uvicorn worker class to run the FastAPI app
gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT