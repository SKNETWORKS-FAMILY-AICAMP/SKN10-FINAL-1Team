
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json
import uuid
from datetime import datetime
import logging
import pandas as pd
import psycopg2
from psycopg2 import sql
from io import StringIO
import tempfile

# Add project root to sys.path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import agent module
from fastapi_server.agent_service import AgentService, get_agent_service
from fastapi_server.models import ChatRequest, ChatResponse, ChatMessage

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Check OpenAI API Key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY or len(OPENAI_API_KEY) < 10:
    raise ValueError("OPENAI_API_KEY is not set or invalid. Please check your .env file.")

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph Agent API",
    description="API for interacting with a multi-agent system built with LangGraph",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active websocket connections and their associated thread IDs
active_connections = {}

@app.on_event("startup")
async def startup_event():
    """Initialize agent service on startup."""
    logger.info("Starting up FastAPI server and initializing LangGraph agent...")
    # Agent service is lazily initialized using dependency


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down FastAPI server...")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "online", "message": "LangGraph Agent API is running"}


# PostgreSQL 연결 설정
def get_db_connection():
    """Get PostgreSQL database connection using credentials from .env"""
    # Load environment variables if not already loaded
    load_dotenv(dotenv_path=dotenv_path)
    
    host = os.environ.get('DB_HOST')
    port = os.environ.get('DB_PORT')
    dbname = os.environ.get('DB_NAME')
    user = os.environ.get('DB_USER')
    password = os.environ.get('DB_PASSWORD')
    
    if not all([host, port, dbname, user, password]):
        raise ValueError("Database configuration is incomplete. Check your .env file.")
    
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    return conn


@app.get("/api/tables")
async def get_tables():
    """Get all available tables from the PostgreSQL database."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Execute query to get all tables in the database
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        tables = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        
        return {"tables": tables}
    except Exception as e:
        logger.error(f"Error fetching tables: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...), table_name: str = Form(...), create_table: bool = Form(False)):
    """Upload a CSV file and save it to the specified PostgreSQL table.
    
    Args:
        file: The CSV file to upload
        table_name: The name of the table to save the data to
        create_table: Whether to create a new table if it doesn't exist
    """
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Read the CSV file with pandas
        df = pd.read_csv(temp_path)
        
        # Remove the temporary file
        os.unlink(temp_path)
        
        # Connect to the PostgreSQL database
        conn = get_db_connection()
        cur = conn.cursor()
        
        # If create_table is True, create the table if it doesn't exist
        if create_table:
            # Generate column definitions from dataframe
            columns = []
            for col_name, dtype in zip(df.columns, df.dtypes):
                if pd.api.types.is_integer_dtype(dtype):
                    col_type = "INTEGER"
                elif pd.api.types.is_float_dtype(dtype):
                    col_type = "FLOAT"
                elif pd.api.types.is_bool_dtype(dtype):
                    col_type = "BOOLEAN"
                elif pd.api.types.is_datetime64_dtype(dtype):
                    col_type = "TIMESTAMP"
                else:
                    col_type = "TEXT"
                
                # Sanitize column name - replace spaces and special chars with underscores
                sanitized_col = ''.join(c if c.isalnum() else '_' for c in col_name)
                columns.append(f"\"{sanitized_col}\" {col_type}")
            
            # Create the table
            create_table_query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                sql.Identifier(table_name),
                sql.SQL(", ").join([sql.SQL(col) for col in columns])
            )
            
            cur.execute(create_table_query)
            conn.commit()
        
        # Create a buffer for the CSV data
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        
        # Generate column names for the COPY command
        columns = [f'"{col}"' for col in df.columns]
        column_str = ", ".join(columns)
        
        # Use the COPY command to efficiently insert the data
        copy_query = f"""COPY {table_name} ({column_str}) FROM STDIN WITH CSV"""
        cur.copy_expert(copy_query, buffer)
        
        # Commit the transaction and close the connection
        conn.commit()
        cur.close()
        conn.close()
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": f"Successfully uploaded {len(df)} rows to table {table_name}",
                "rows_uploaded": len(df)
            }
        )
        
    except Exception as e:
        logger.error(f"Error uploading CSV: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


from pydantic import BaseModel

class PredictionRequest(BaseModel):
    table_name: str
    model_type: str = "churn"

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Run prediction on data from a specified table using the ML model.
    
    The model will handle the preprocessing of raw data before prediction.
    
    Args:
        request: PredictionRequest object containing table_name and model_type
    """
    try:
        # Connect to the database
        conn = get_db_connection()
        
        # Read data from the specified table
        df = pd.read_sql(f"SELECT * FROM {request.table_name}", conn)
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found in table {request.table_name}")
        
        # Get the agent service to process the prediction
        agent_service = get_agent_service()
        
        # Process the prediction (the agent service will handle preprocessing and prediction)
        prediction_results = await agent_service.process_prediction(df, request.model_type)
        
        return {"predictions": prediction_results.to_dict(orient="records")}
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")



@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Process a chat message through the LangGraph agent system.
    """
    try:
        # Generate a thread ID if not provided
        thread_id = request.thread_id or f"thread_{uuid.uuid4()}"
        
        # Process the request through the agent service
        response = await agent_service.process_message(
            request.message, 
            thread_id=thread_id
        )
        
        return ChatResponse(
            thread_id=thread_id,
            message_id=str(uuid.uuid4()),
            content=response["content"],
            created_at=datetime.now().isoformat(),
            role="assistant",
            agent=response.get("agent", "system")
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent processing error: {str(e)}")


@app.websocket("/api/chat/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str, agent_service: AgentService = Depends(get_agent_service)):
    """
    WebSocket endpoint for streaming chat interactions with the LangGraph agent.
    """
    await websocket.accept()
    
    # Register this connection
    connection_id = str(uuid.uuid4())
    active_connections[connection_id] = {"websocket": websocket, "thread_id": thread_id}
    
    try:
        logger.info(f"WebSocket connection established: {connection_id} for thread {thread_id}")
        
        while True:
            # Receive and parse the message
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                
                if not user_message:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Invalid message format. Expected 'message' field."
                    })
                    continue
                
                # Process through agent with streaming
                await agent_service.process_message_stream(
                    user_message,
                    thread_id=thread_id,
                    websocket=websocket
                )
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid JSON format"
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"Server error: {str(e)}"
            })
        except:
            pass
    finally:
        # Clean up on disconnect
        if connection_id in active_connections:
            del active_connections[connection_id]
        logger.info(f"WebSocket connection closed: {connection_id}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
