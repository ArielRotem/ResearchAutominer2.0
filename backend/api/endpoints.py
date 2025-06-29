
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from backend.core.function_loader import load_functions
from backend.core.manuscript_runner import run_manuscript
from backend.api.gemini_client import generate_function
import pandas as pd
import numpy as np
import io
import os
import json
from typing import List, Dict, Any

router = APIRouter()
MANUSCRIPTS_DIR = "manuscripts"

@router.get("/functions")
def list_functions():
    """Returns a list of all available data processing functions."""
    functions = load_functions()
    return {
        name: {
            "name": data["name"],
            "doc": data["doc"],
            "params": data["params"]
        }
        for name, data in functions.items()
    }

@router.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """Uploads a CSV file and returns its headers and first 50 rows."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a .csv")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), low_memory=False) # Use low_memory=False to suppress DtypeWarning and improve type inference
        
        # Robustly handle non-JSON-serializable values (NaN, inf, -inf) and convert all to string for JSON compatibility
        for col in df.columns:
            df[col] = df[col].astype(str) # Convert entire column to string
            df[col] = df[col].replace('nan', None) # Replace string 'nan' with None
            df[col] = df[col].replace('inf', None) # Replace string 'inf' with None
            df[col] = df[col].replace('-inf', None) # Replace string '-inf' with None

        headers = df.columns.tolist()
        all_data = df.to_dict(orient="records") # Send all data for frontend pagination
        
        return {
            "filename": file.filename,
            "headers": headers,
            "rowCount": len(df),
            "allData": all_data # Renamed from sampleData to allData
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@router.post("/test_manuscript")
async def test_manuscript(data: Dict[str, Any] = Body(...)):
    """Runs a manuscript on a small sample of data and returns the result."""
    sample_data = data.get("sample_data", [])
    manuscript = data.get("manuscript", [])
    
    if not sample_data or not manuscript:
        raise HTTPException(status_code=400, detail="sample_data and manuscript are required.")

    try:
        df = pd.DataFrame(sample_data)
        result_df = run_manuscript(df, manuscript)
        return result_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing manuscript: {e}\n\nTraceback:\n{traceback.format_exc()}")

@router.post("/run_full_manuscript")
async def run_full_manuscript(payload: Dict[str, Any] = Body(...)):
    """Runs a full manuscript on the provided data and returns the processed data."""
    data_to_process = payload.get("data", [])
    manuscript = payload.get("manuscript", [])

    if not data_to_process or not manuscript:
        raise HTTPException(status_code=400, detail="Data and manuscript are required.")

    try:
        df = pd.DataFrame(data_to_process)
        processed_df = run_manuscript(df, manuscript)
        
        # Convert all columns to string type to ensure JSON compatibility
        for col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(str) # Convert entire column to string
            processed_df[col] = processed_df[col].replace('nan', None) # Replace string 'nan' with None
            processed_df[col] = processed_df[col].replace('inf', None) # Replace string 'inf' with None
            processed_df[col] = processed_df[col].replace('-inf', None) # Replace string '-inf' with None

        return processed_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running full manuscript: {e}\n\nTraceback:\n{traceback.format_exc()}")

@router.get("/manuscripts")
def list_manuscripts():
    """Lists all saved manuscript files."""
    if not os.path.exists(MANUSCRIPTS_DIR):
        return []
    return [f for f in os.listdir(MANUSCRIPTS_DIR) if f.endswith('.json')]

@router.post("/manuscripts/{filename}")
async def save_manuscript(filename: str, manuscript: List[Dict[str, Any]] = Body(...)):
    """Saves a manuscript to a file."""
    if not filename.endswith('.json'):
        filename += '.json'
    
    if not os.path.exists(MANUSCRIPTS_DIR):
        os.makedirs(MANUSCRIPTS_DIR)
        
    filepath = os.path.join(MANUSCRIPTS_DIR, filename)
    try:
        with open(filepath, 'w') as f:
            json.dump(manuscript, f, indent=4)
        return {"message": f"Manuscript '{filename}' saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/manuscripts/{filename}")
def load_manuscript(filename: str):
    """Loads a manuscript from a file."""
    filepath = os.path.join(MANUSCRIPTS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Manuscript not found.")
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_function")
async def generate_function_endpoint(data: Dict[str, Any] = Body(...)):
    """Generates a Python function using the Gemini API."""
    prompt = data.get("prompt")
    headers = data.get("headers")
    if not prompt or not headers:
        raise HTTPException(status_code=400, detail="prompt and headers are required.")
    
    try:
        generated_code = generate_function(prompt, headers)
        return {"code": generated_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
