#!/bin/sh
# This script is called by the webhook to trigger a restart.

echo "Redeployment triggered. Shutting down in 3 seconds to allow the webhook response to be sent."
sleep 3

# Kill the main process (PID 1), which is uvicorn.
# RunPod should then restart the pod, pulling the latest image.
kill 1
