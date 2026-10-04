#!/usr/bin/env bash
# Starts the web app on port 8000. Run from anywhere:  bash start_server.sh
set -e
cd "$(dirname "$0")/backend"
uvicorn main:app --host 0.0.0.0 --port 8000
