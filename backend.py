# backend.py
import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Add the 'app' directory to the Python path so we can import main
# Adjust if your structure is different or use relative imports if preferred
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

# Import the function AFTER modifying sys.path
try:
    from main import sql_copilot
except ImportError as e:
    print(f"Error importing sql_copilot from app.main: {e}")
    print(f"Current sys.path: {sys.path}")
    # Define a dummy function to allow FastAPI to start, but it will fail at runtime
    def sql_copilot(user_question: str):
        return f"Error: Could not load the core application logic (sql_copilot function). Check backend logs and imports. ImportError: {e}"

# Define the request body model
class QueryRequest(BaseModel):
    question: str

# Create the FastAPI app instance
app = FastAPI()

@app.post("/query")
async def run_query(request: QueryRequest):
    """
    Receives a question, processes it using the sql_copilot function,
    and returns the result.
    """
    print(f"Backend received query: {request.question}")
    try:
        result = sql_copilot(user_question=request.question)
        print("Backend processing complete.")
        return {"result": result}
    except Exception as e:
        print(f"Error during backend processing: {e}")
        # You might want more specific error handling here
        return {"result": f"An error occurred on the backend: {e}"}

@app.get("/")
async def read_root():
    return {"message": "SQL Copilot Backend is running. POST to /query with {'question': 'your question'}"}

# --- How to run the backend ---
# Open a terminal in the project root directory and run:
# uvicorn backend:app --reload --host 0.0.0.0 --port 8000
#
# --reload: Automatically restarts the server when code changes (good for development)
# --host 0.0.0.0: Makes the server accessible on your network
# --port 8000: Specifies the port number
#
# Keep this terminal window open while the backend is running.
# -----------------------------

if __name__ == "__main__":
    # This allows running directly with `python backend.py` for simple testing,
    # but `uvicorn` is recommended for development and production.
    uvicorn.run(app, host="0.0.0.0", port=8000)