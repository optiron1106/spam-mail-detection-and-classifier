from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .model_service import get_service, PROJECT_ROOT

FRONTEND_BUILD_DIR = PROJECT_ROOT / "frontend" / "dist"

app = FastAPI(title="Spam Detection API", version="1.0.0")

# CORS (open by default; adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


@app.on_event("startup")
def startup_event() -> None:
    get_service().ensure_artifacts()


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
def predict(req: PredictRequest) -> JSONResponse:
    try:
        result = get_service().predict(req.text)
        return JSONResponse(content=result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# If a React production build exists, serve it; otherwise fall back to static HTML
if (FRONTEND_BUILD_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def index() -> FileResponse:
        static_index = PROJECT_ROOT / "static" / "index.html"
        return FileResponse(static_index)
