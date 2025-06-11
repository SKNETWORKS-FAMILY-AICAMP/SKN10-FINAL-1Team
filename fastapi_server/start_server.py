"""
A simple script to start the FastAPI server using uvicorn.
This can be used as a convenient way to start the server from a command line.
"""

import uvicorn

if __name__ == "__main__":
    print("Starting LangGraph Agent FastAPI Server on port 8001...")
    uvicorn.run("fastapi_server.main:app", host="0.0.0.0", port=8001, reload=True)
