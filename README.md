# 1to100_Korean

# Korean Language Problem Extraction and Classification Service

This project aims to automatically extract and classify Korean language problems from documents using the LayoutLMv3 model.

## Project Structure

-   `src/`: Main source code for the service.
-   `data/`: Datasets for training and evaluation.
-   `models/`: Trained models.
-   `notebooks/`: Jupyter notebooks for experimentation.
-   `tests/`: Tests for the codebase.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the web server:
    ```bash
    uvicorn src.web_server:app --reload
    ```

## Usage

### CLI

Run the full pipeline from the command line:

```bash
python -m src.main --pdf <PDF_파일경로>
```

### Web Server

Start the FastAPI server and upload a PDF via browser:

```bash
uvicorn src.web_server:app --reload
```

Then open `http://localhost:8000` and upload a PDF to receive the shuffled result.

