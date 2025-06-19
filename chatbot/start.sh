#!/bin/bash
set -e

echo "[SYSTEM] Starting services..."

# Start the API server with tagged output
echo "[SYSTEM] Starting API server..."
python api.py 2>&1 | sed 's/^/[API] /' > /proc/1/fd/1 &
API_PID=$!

# Wait a moment to ensure API starts up
sleep 2
echo "[SYSTEM] API server started with PID: $API_PID"

# Start Streamlit with tagged output
echo "[SYSTEM] Starting Streamlit server..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 2>&1 | sed 's/^/[UI] /'

# If Streamlit exits, kill the API process
kill $API_PID