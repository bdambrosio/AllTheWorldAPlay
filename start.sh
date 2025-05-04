#!/bin/bash
# Start the Python backend
python src/sim/main.py &
# Wait a moment for the backend to initialize
sleep 5
# Start the frontend development server if it exists
if [ -d "/app/src/sim/webworld" ]; then
  cd /app/src/sim/webworld
  if [ -f "package.json" ]; then
    npm start
  fi
fi
# Keep container running
wait
