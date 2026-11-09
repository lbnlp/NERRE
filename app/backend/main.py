"""HTTP API for local document extraction."""
import io
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pypdf import PdfReader

from app.backend.extractor import DEFAULT_MODEL, Extractor, split_sentences
from app.backend.schemas import ExtractionResponse

logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.environ.get("NERRE_MODEL_PATH", str(DEFAULT_MODEL)))
MAX_BYTES = 5 * 1024 * 1024
MAX_SENTENCES = 100
MAX_PDF_PAGES = 30
app = FastAPI(title="NERRE document extraction")


class UploadLimitMiddleware:
    """Bound the multipart request before the parser can spool it to disk."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST' or scope['path'] != '/api/extract':
            return await self.app(scope, receive, send)
        limit = MAX_BYTES + 64 * 1024  # Allow multipart headers in addition to the file.
        headers = dict(scope.get('headers', []))
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            length = 0
        if length > limit:
            return await JSONResponse({'detail': 'File exceeds the 5 MB limit.'}, status_code=413)(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            total += len(message.get('body', b''))
            if total > limit:
                return await JSONResponse({'detail': 'File exceeds the 5 MB limit.'}, status_code=413)(scope, receive, send)
            chunks.append(message)
            if not message.get('more_body', False):
                break
        iterator = iter(chunks)
        async def replay():
            return next(iterator, {'type': 'http.disconnect'})
        return await self.app(scope, replay, send)


app.add_middleware(UploadLimitMiddleware)
_extractor = None
_loading_lock = threading.Lock()
_generation_lock = threading.Lock()


def get_extractor():
    global _extractor
    with _loading_lock:
        if _extractor is None:
            if not (MODEL_PATH / "config.json").is_file():
                raise HTTPException(503, "No fine-tuned model found. Run the training command in the README first.")
            try:
                _extractor = Extractor(MODEL_PATH)
            except Exception as exc:
                logger.exception("Failed to load model from %s", MODEL_PATH)
                raise HTTPException(503, "Could not load the fine-tuned model. Check the server logs and model path.") from exc
        return _extractor


@app.get("/")
def index():
    return {"message": "NERRE extraction API", "status": "/api/status"}


@app.get("/api/status")
def status():
    return {"model_ready": (MODEL_PATH / "config.json").is_file(), "task": "doping"}


def read_document(name, content):
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(400, "Text files must be UTF-8 encoded.") from exc
    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise HTTPException(400, "Encrypted PDFs are not supported.")
            if len(reader.pages) > MAX_PDF_PAGES:
                raise HTTPException(400, f"PDF exceeds the {MAX_PDF_PAGES}-page limit.")
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, "Could not read this PDF.") from exc
    raise HTTPException(400, "Upload a PDF or UTF-8 text file.")


@app.post("/api/extract", response_model=ExtractionResponse)
def extract(file: UploadFile = File(...)):
    content = file.file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "File exceeds the 5 MB limit.")
    text = read_document(file.filename or "", content)
    sentences = split_sentences(text)
    if not sentences:
        raise HTTPException(400, "No text found. Scanned PDFs need OCR before upload.")
    if len(sentences) > MAX_SENTENCES:
        raise HTTPException(400, f"Document exceeds the {MAX_SENTENCES}-sentence limit. Upload a shorter document.")
    model = get_extractor()
    results = []
    # Model generation is serialized to avoid concurrent CPU/memory spikes.
    with _generation_lock:
        for sentence in sentences:
            try:
                result = model.extract(sentence)
            except ValueError as exc:
                results.append({"sentence": sentence, "status": "error", "error": str(exc)})
            else:
                results.append({"sentence": sentence, "status": "complete", **result})
    status = "partial" if any(result["status"] == "error" for result in results) else "complete"
    return {"filename": Path(file.filename or "document").name, "schema": "doping", "status": status, "sentences": results}
