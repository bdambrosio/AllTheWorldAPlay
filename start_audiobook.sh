#!/bin/bash
# Start the Python backend
cd /app/src/sim/
uvicorn replay:app --host 0.0.0.0 --port 8000 &      
cd /app/src/sim/audiobook
npm install
npm start &                                                    # UI
wait -n                                     # exit if any child dies
