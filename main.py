import io
import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import easyocr
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Logging
logger = logging.getLogger("batch-ocr")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# FastAPI app
app = FastAPI(title="Batch OCR System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# EasyOCR reader
_reader = None
def get_reader(language: str = "eng"):
    global _reader
    if _reader is None:
        logger.info(f"Initializing EasyOCR reader for language: {language}")
        _reader = easyocr.Reader([language], gpu=False)
        logger.info("EasyOCR ready.")
    return _reader

# Helpers
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}

def clean_text(raw: str) -> str:
    text = raw.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()

def validate_files(files: List[UploadFile]) -> bool:
    from pathlib import Path
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False
    return True

def ocr_image(image_bytes: bytes, language: str = "eng") -> str:
    reader = get_reader(language)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image.")
    result = reader.readtext(img, detail=0, paragraph=True)
    return " ".join(result)

def process_single_file(filename: str, contents: bytes, language: str):
    try:
        raw = ocr_image(contents, language)
        text = clean_text(raw)
        logger.info(f"Success: {filename}")
        return filename, text, ""
    except Exception as e:
        err = str(e)
        logger.error(f"Error processing {filename}: {err}")
        return filename, "", err

# Routes
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    language: str = Form("eng"),
):
    if not validate_files(files):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {}
        for file in files:
            contents = await file.read()
            future = executor.submit(process_single_file, file.filename, contents, language)
            future_to_file[future] = file.filename

        for future in as_completed(future_to_file):
            filename, text, error = future.result()
            results.append({"filename": filename, "text": text, "error": error})

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["filename", "text"])
    for row in results:
        writer.writerow([row["filename"], row["text"]])

    csv_buffer.seek(0)
    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ocr_results.csv"},
    )

# Uvicorn entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
