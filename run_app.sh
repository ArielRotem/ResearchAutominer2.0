#!/bin/bash

# Exit on error
set -e

# Function to kill all background processes on exit
trap 'trap - SIGTERM && kill 0' SIGINT SIGTERM EXIT

# Backend setup
echo "Setting up and starting backend..."
cd backend

# Install dependencies
python3 -m pip install --quiet -r requirements.txt

cd ..

# Start the FastAPI server in the background
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait a moment for the server to start
sleep 3

cd ..

# Frontend setup
echo "Setting up and starting frontend..."
cd frontend

# Check if node_modules exists, if not, run npm install
if [ ! -d "node_modules" ]; then
  echo "Node modules not found, running npm install..."
  npm install
fi

# Start the React development server
npm start
