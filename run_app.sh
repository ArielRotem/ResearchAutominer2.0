#!/bin/bash

# Exit on error
set -e

# Function to kill all background processes on exit
trap 'trap - SIGTERM && kill 0' SIGINT SIGTERM EXIT

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Backend setup
echo "Setting up and starting backend..."
cd "$SCRIPT_DIR/backend"

# Install dependencies
python3 -m pip install --quiet -r requirements.txt

# Go back to the script's directory
cd "$SCRIPT_DIR"

# Start the FastAPI server in the background
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait a moment for the server to start
sleep 3

# Frontend setup
echo "Setting up and starting frontend..."
cd "$SCRIPT_DIR/frontend"

# Check if node_modules exists, if not, run npm install
if [ ! -d "node_modules" ]; then
  echo "Node modules not found, running npm install..."
  npm install
fi

# Start the React development server
npm start