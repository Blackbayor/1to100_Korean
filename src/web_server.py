import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from src.main import run_pipeline
from src.config import Config

app = FastAPI()
config = Config()
templates = Jinja2Templates(directory="templates")

# --- Directories ---
results_dir = os.path.join(config.PROJECT_ROOT, "results")
history_dir = os.path.join(config.PROJECT_ROOT, "history")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

# --- Static Files ---
app.mount("/results", StaticFiles(directory=results_dir), name="results")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Renders the main page with history and results."""
    history_files = os.listdir(history_dir)
    result_files = os.listdir(results_dir)
    return templates.TemplateResponse("index.html", {"request": request, "history_files": history_files, "result_files": result_files})

@app.post("/shuffle")
async def shuffle_pdf(request: Request, file: UploadFile = File(...)):
    """Handles PDF upload, shuffling, and redirects to the main page."""
    history_path = os.path.join(history_dir, file.filename)
    with open(history_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    await shuffle_from_path(request, history_path)
    return RedirectResponse(url="/", status_code=303)

@app.post("/shuffle-history/{filename}")
async def shuffle_history_pdf(request: Request, filename: str):
    """Handles shuffling from a file in the history and redirects to the main page."""
    history_path = os.path.join(history_dir, filename)
    if not os.path.exists(history_path):
        raise HTTPException(status_code=404, detail="File not found in history.")

    await shuffle_from_path(request, history_path)
    return RedirectResponse(url="/", status_code=303)

@app.get("/view-result/{filename}")
async def view_result(request: Request, filename: str):
    """Displays the result PDF in a viewer."""
    pdf_url = request.url_for("results", path=filename)
    return templates.TemplateResponse("result.html", {"request": request, "pdf_url": pdf_url})

async def shuffle_from_path(request: Request, file_path: str):
    """Common shuffling logic."""
    request_id = str(uuid.uuid4())
    upload_dir = os.path.join(config.PROJECT_ROOT, "uploads", request_id)
    os.makedirs(upload_dir, exist_ok=True)

    saved_path = os.path.join(upload_dir, os.path.basename(file_path))
    shutil.copy(file_path, saved_path)

    try:
        output_pdf_path = await run_in_threadpool(run_pipeline, saved_path, request_id)

        result_filename = f"{request_id}_{os.path.basename(output_pdf_path)}"
        result_path = os.path.join(results_dir, result_filename)
        shutil.move(output_pdf_path, result_path)

    except Exception as e:
        print(f"Error processing file: {e}")
        # In a real app, you might want to handle this more gracefully
        # For now, we'll just print the error and not re-raise
    finally:
        shutil.rmtree(upload_dir)