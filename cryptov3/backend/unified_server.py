from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import importlib.util
from typing import Optional

# Create a new main application
app = FastAPI(
    title="Candlestick Chart Unified API",
    description="Unified API for candlestick chart prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to safely import a module from a file path
def import_module_from_file(module_name, file_path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"Failed to load spec for {file_path}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing {file_path}: {str(e)}")
        return None

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the main app endpoints
main_path = os.path.join(current_dir, "main.py")
if os.path.exists(main_path):
    print(f"Importing endpoints from {main_path}")
    main_module = import_module_from_file("main_module", main_path)
    if main_module and hasattr(main_module, "app"):
        print("Adding routes from main.py")
        # Copy all routes from main_app to our app
        for route in main_module.app.routes:
            app.routes.append(route)
    else:
        print("Failed to import main.py")

# Import the app.py endpoints
app_path = os.path.join(current_dir, "app.py")
if os.path.exists(app_path):
    print(f"Importing endpoints from {app_path}")
    app_module = import_module_from_file("app_module", app_path)
    if app_module and hasattr(app_module, "app"):
        print("Adding routes from app.py")
        # Copy all routes from app_app to our app
        for route in app_module.app.routes:
            app.routes.append(route)
    else:
        print("Failed to import app.py")

# Add a root route
@app.get("/")
async def root():
    return {
        "status": "Unified API is running",
        "message": "Welcome to the unified Candlestick Chart API",
        "available_endpoints": [
            "/api/predict",
            "/api/imgNewModel",
            "/api/health"
        ]
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting unified server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
