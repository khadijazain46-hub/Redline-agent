import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from .models import ReviewRequest, ReviewResponse, Issue
from .storage import UPLOADS_DIR, OUTPUTS_DIR, PREVIEWS_DIR, saved_path
from .parsers import parse_dxf
from .rules import static_checks
from .ai import ai_checks
from .redline import annotate_issues
from .render import render_preview

load_dotenv()

app = FastAPI(title="Redline Agent")

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Serve static front-end and generated files
# app.mount("/static", StaticFiles(directory=str((os.path.dirname(__file__)) + "/../static")), name="static")
# app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
# app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
# app.mount("/previews", StaticFiles(directory=str(PREVIEWS_DIR)), name="previews")


@app.get("/")
def root():
    return {"message": "Redline Agent API", "upload": "/api/upload", "review": "/api/review"}


# ---- React-compatible upload endpoint ----
@app.post("/api/upload")
async def upload(file: UploadFile = File(...), ai: bool = Form(True)):
    name = file.filename
    if not name.lower().endswith(".dxf"):
        raise HTTPException(400, detail="Only .dxf is supported in MVP. Convert .dwg to .dxf first.")
    
    # Save uploaded file
    path = saved_path(UPLOADS_DIR, name)
    with open(path, "wb") as f:
        f.write(await file.read())

    # Parse DXF
    parsed = parse_dxf(str(path))

    # Run AI if requested
    issues: List[Issue] = ai_checks(parsed.entities) if ai else []

    # Annotate DXF
    out_name = name.replace(".dxf", "_REDLINES.dxf")
    dxf_out = saved_path(OUTPUTS_DIR, out_name)
    annotate_issues(str(path), str(dxf_out), issues)

    # Optional previews
    png_out = saved_path(PREVIEWS_DIR, out_name.replace(".dxf", ".png"))
    pdf_out = saved_path(PREVIEWS_DIR, out_name.replace(".dxf", ".pdf"))
    try:
        render_preview(str(dxf_out), str(png_out), str(pdf_out))
        png_url = f"/previews/{png_out.name}"
        pdf_url = f"/previews/{pdf_out.name}"
    except Exception:
        png_url = None
        pdf_url = None

    # Return directly for React download
    return FileResponse(
        path=str(dxf_out),
        filename=out_name,
        media_type="application/dxf"
    )


# ---- Full review endpoint (optional) ----
@app.post("/api/review", response_model=ReviewResponse)
async def review(req: ReviewRequest):
    dxf_in = saved_path(UPLOADS_DIR, req.file_name)
    if not dxf_in.exists():
        raise HTTPException(404, detail="File not found. Upload first.")

    parsed = parse_dxf(str(dxf_in))

    issues: List[Issue] = []
    if req.static_rules:
        issues.extend(static_checks(parsed.entities))
    if req.ai:
        issues.extend(ai_checks(parsed.entities))

    # Annotate
    out_name = req.file_name.replace(".dxf", "_REDLINES.dxf")
    dxf_out = saved_path(OUTPUTS_DIR, out_name)
    annotate_issues(str(dxf_in), str(dxf_out), issues)

    # Previews
    png_out = saved_path(PREVIEWS_DIR, out_name.replace(".dxf", ".png"))
    pdf_out = saved_path(PREVIEWS_DIR, out_name.replace(".dxf", ".pdf"))
    try:
        render_preview(str(dxf_out), str(png_out), str(pdf_out))
        png_url = f"/previews/{png_out.name}"
        pdf_url = f"/previews/{pdf_out.name}"
    except Exception:
        png_url = None
        pdf_url = None

    return ReviewResponse(
        file_name=req.file_name,
        issues=issues,
        annotated_dxf=f"/outputs/{dxf_out.name}",
        preview_png=png_url,
        preview_pdf=pdf_url,
    )


@app.get("/api/download/annotated/{file_name}")
async def download_annotated(file_name: str):
    path = saved_path(OUTPUTS_DIR, file_name)
    if not path.exists():
        raise HTTPException(404, detail="Not found")
    return FileResponse(str(path), filename=file_name)
