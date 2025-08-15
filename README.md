# Redline Agent – Full Code (FastAPI + AI + DXF Redlines)

A production-ready MVP that:

* Accepts `.dxf` (and optionally `.dwg` via conversion)
* Parses geometry, text, dimensions
* Runs **static checks** + **AI checks**
* Generates **redline annotations** on a `REDLINES` layer
* Exports **annotated DXF** and a **PDF/PNG preview**
* Simple HTML front-end uploader with results

---

## File Tree

```
redline-agent/
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ models.py
│  ├─ storage.py
│  ├─ parsers.py
│  ├─ rules.py
│  ├─ ai.py
│  ├─ redline.py
│  ├─ render.py
│  └─ utils.py
├─ static/
│  └─ index.html
├─ .env.example
├─ requirements.txt
├─ README.md
└─ Dockerfile
```

---

## requirements.txt

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.5
ezdxf==1.3.4
pydantic==2.9.0
python-multipart==0.0.9
matplotlib==3.9.1
shapely==2.0.5
# AI providers (optional – install only the one you use)
openai==1.43.0
groq==0.11.0
python-dotenv==1.0.1
```

> **Notes**
>
> * `matplotlib` is used by `ezdxf.addons.drawing` to render PNG/PDF previews.
> * `shapely` helps with robust geometry (point-in-polygon, buffers, etc.).

---

## .env.example

```env
# Choose: openai | groq | none
AI_PROVIDER=none
# If using OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
# If using Groq
GROQ_API_KEY=gk-...
GROQ_MODEL=llama3-70b-8192

# Storage
DATA_DIR=./data
```

---

## app/**init**.py

```python
# empty init to make "app" a package
```

---

## app/models.py

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any

class Issue(BaseModel):
    x: float
    y: float
    issue: str
    meta: Dict[str, Any] = Field(default_factory=dict)

class ReviewRequest(BaseModel):
    file_name: str
    ai: bool = True
    static_rules: bool = True

class ReviewResponse(BaseModel):
    file_name: str
    issues: List[Issue]
    annotated_dxf: Optional[str] = None
    preview_pdf: Optional[str] = None
    preview_png: Optional[str] = None

class Entity(BaseModel):
    type: str
    layer: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    text: Optional[str] = None
    insert: Optional[Tuple[float, float]] = None
    points: Optional[List[Tuple[float, float]]] = None

class ParseResult(BaseModel):
    entities: List[Entity]
```

---

## app/storage.py

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
PREVIEWS_DIR = DATA_DIR / "previews"

for d in (DATA_DIR, UPLOADS_DIR, OUTPUTS_DIR, PREVIEWS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def saved_path(dirpath: Path, filename: str) -> Path:
    return dirpath / filename
```

---

## app/utils.py

```python
from typing import Iterable, Tuple

BBox = Tuple[float, float, float, float]


def union_bbox(a: BBox, b: BBox) -> BBox:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))


def centroid_from_bbox(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_from_points(points: Iterable[Tuple[float, float]]) -> BBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))
```

---

## app/parsers.py

```python
from typing import List, Tuple
import ezdxf
from ezdxf.entities import DXFGraphic
from shapely.geometry import Point, Polygon
from .models import Entity, ParseResult
from .utils import bbox_from_points


def _entity_bbox(e: DXFGraphic):
    try:
        if e.dxftype() in {"LWPOLYLINE"}:
            pts = [(p[0], p[1]) for p in e.get_points()]
            return bbox_from_points(pts)
        if e.dxftype() in {"LINE"}:
            p1 = (float(e.dxf.start.x), float(e.dxf.start.y))
            p2 = (float(e.dxf.end.x), float(e.dxf.end.y))
            return bbox_from_points([p1, p2])
        if hasattr(e, "bbox") and e.bbox():
            b = e.bbox()
            # ezdxf bbox returns (extmin, extmax)
            (x1, y1, _), (x2, y2, _) = b.extmin, b.extmax
            return (x1, y1, x2, y2)
    except Exception:
        pass
    return None


def parse_dxf(dxf_path: str) -> ParseResult:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    entities: List[Entity] = []
    for e in msp:
        et = e.dxftype()
        layer = getattr(e.dxf, "layer", "")
        bbox = _entity_bbox(e)
        insert = None
        text = None
        points = None

        if et in {"TEXT", "MTEXT"}:
            try:
                text = e.dxf.text if hasattr(e.dxf, "text") else e.plain_text()
            except Exception:
                text = None
            try:
                insert = (float(e.dxf.insert.x), float(e.dxf.insert.y))
            except Exception:
                insert = None
        elif et == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            points = pts
        elif et == "CIRCLE":
            try:
                insert = (float(e.dxf.center.x), float(e.dxf.center.y))
            except Exception:
                insert = None
        elif et == "ARC":
            try:
                insert = (float(e.dxf.center.x), float(e.dxf.center.y))
            except Exception:
                insert = None

        entities.append(Entity(type=et, layer=layer, bbox=bbox, text=text, insert=insert, points=points))

    return ParseResult(entities=entities)


def polygon_contains_text(room_points: List[Tuple[float, float]], texts: List[Entity]) -> bool:
    poly = Polygon(room_points)
    for t in texts:
        if t.insert is None:
            continue
        if poly.contains(Point(t.insert[0], t.insert[1])):
            return True
    return False
```

---


## app/rules.py

```python
from typing import List
from .models import Entity, Issue
from .parsers import polygon_contains_text


def static_checks(entities: List[Entity]) -> List[Issue]:
    """Simple deterministic checks as examples.
    - Rooms drawn as LWPOLYLINE must contain a TEXT/MTEXT label.
    - Warn if any TEXT is on layer '0' (encourage labeling layers).
    """
    issues: List[Issue] = []

    rooms = [e for e in entities if e.type == "LWPOLYLINE" and e.points]
    texts = [e for e in entities if e.type in ("TEXT", "MTEXT")]

    # Check 1: missing room label
    for r in rooms:
        if not polygon_contains_text(r.points, texts):
            # Place marker at polygon centroid (approx via bbox)
            if r.bbox:
                x = (r.bbox[0] + r.bbox[2]) / 2
                y = (r.bbox[1] + r.bbox[3]) / 2
            else:
                x = r.points[0][0]
                y = r.points[0][1]
            issues.append(Issue(x=x, y=y, issue="Missing room label", meta={"rule": "ROOM_LABEL_REQUIRED"}))

    # Check 2: text on layer 0
    for t in texts:
        if t.layer.strip() == "0":
            ix, iy = (t.insert or (t.bbox[0], t.bbox[1]) if t.bbox else (0.0, 0.0))
            issues.append(Issue(x=ix, y=iy, issue="Text on default layer '0'", meta={"rule": "TEXT_NOT_ON_LAYER_0"}))

    return issues
```

---

## app/ai.py

```python
import os
import json
from typing import List
from dotenv import load_dotenv
from .models import Entity, Issue

load_dotenv()

AI_PROVIDER = os.getenv("AI_PROVIDER", "none").lower()

# Optional providers
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

try:
    from groq import Groq as _Groq
except Exception:
    _Groq = None


def summarize_entities_for_llm(entities: List[Entity], limit: int = 300) -> str:
    """Compact summary to keep token usage sane."""
    rows = []
    for e in entities[:limit]:
        row = {
            "type": e.type,
            "layer": e.layer,
            "insert": e.insert,
            "bbox": e.bbox,
        }
        if e.text:
            row["text"] = e.text[:80]
        if e.points and len(e.points) <= 20:
            row["points"] = e.points
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False)


def ai_checks(entities: List[Entity]) -> List[Issue]:
    if AI_PROVIDER == "none":
        return []

    summary = summarize_entities_for_llm(entities)

    system_prompt = (
        "You are a senior CAD reviewer. Given a structured list of entities from a DXF, "
        "find likely drafting issues. Prefer high-precision items (labels, obvious dimensions). "
        "Return ONLY a JSON array of objects with keys x, y, issue, meta."
    )

    user_prompt = f"Entities (truncated):\n{summary}\n\nRules of thumb: rooms must have labels; doors ≥ 800mm clear; no text on layer '0'."

    if AI_PROVIDER == "openai" and _OpenAI:
        client = _OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content
    elif AI_PROVIDER == "groq" and _Groq:
        client = _Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        # library returns dict-like objects
        content = resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) else resp.choices[0].message.content
    else:
        # Fallback no-op if provider not available
        return []

    try:
        raw = json.loads(content)
        issues: List[Issue] = []
        for it in raw:
            issues.append(Issue(x=float(it.get("x", 0.0)), y=float(it.get("y", 0.0)), issue=str(it.get("issue", "AI-flagged")), meta=it.get("meta", {})))
        return issues
    except Exception:
        # If LLM didn't return pure JSON, ignore AI output gracefully
        return []
```

---

## app/redline.py

```python
import ezdxf
from typing import List
from .models import Issue


def ensure_redline_layer(doc: ezdxf.EzDxf) -> None:
    if "REDLINES" not in doc.layers:
        doc.layers.add("REDLINES", color=1)  # 1 = red


def annotate_issues(dxf_path_in: str, dxf_path_out: str, issues: List[Issue]) -> None:
    doc = ezdxf.readfile(dxf_path_in)
    msp = doc.modelspace()
    ensure_redline_layer(doc)

    for it in issues:
        x, y = float(it.x), float(it.y)
        msp.add_circle((x, y), radius=200.0, dxfattribs={"layer": "REDLINES"})
        msp.add_text(it.issue, dxfattribs={"insert": (x + 250.0, y + 250.0), "layer": "REDLINES"})

    doc.saveas(dxf_path_out)
```

---

## app/render.py

```python
from pathlib import Path
import ezdxf
from ezdxf.addons.drawing import matplotlib as ezdxf_matplotlib
from ezdxf.addons.drawing.config import Configuration


def render_preview(dxf_path: str, png_out: str, pdf_out: str):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    config = Configuration()  # default display config

    # PNG
    ctx = ezdxf_matplotlib.MatplotlibBackend(dpi=200)
    ezdxf_matplotlib.draw(msp, ctx, config)
    fig = ctx.get_figure()
    fig.savefig(png_out)
    fig.clf()

    # PDF
    ctx2 = ezdxf_matplotlib.MatplotlibBackend(dpi=200)
    ezdxf_matplotlib.draw(msp, ctx2, config)
    fig2 = ctx2.get_figure()
    fig2.savefig(pdf_out)
    fig2.clf()
```

---

## app/main.py

```python
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
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

# Allow local dev + any origin (tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"];
    allow_headers=["*"],
)

# Serve static front-end and generated files
app.mount("/static", StaticFiles(directory=str((os.path.dirname(__file__)) + "/../static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/previews", StaticFiles(directory=str(PREVIEWS_DIR)), name="previews")


@app.get("/")
def root():
    # Simple landing page
    return {"message": "Redline Agent API", "upload": "/api/upload", "review": "/api/review"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename
    if not name.lower().endswith(".dxf"):
        raise HTTPException(400, detail="Only .dxf is supported in MVP. Convert .dwg to .dxf first.")
    path = saved_path(UPLOADS_DIR, name)
    with open(path, "wb") as f:
        data = await file.read()
        f.write(data)
    return {"file_name": name, "path": str(path)}


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
```

> **Tip:** To support `.dwg`, add a converter step (ODA File Converter CLI) before `parse_dxf()` and pass the converted `.dxf` path.

---

## static/index.html

```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Redline Agent</title>
    <style>
      body{font-family:ui-sans-serif,system-ui,Arial;margin:40px;}
      .card{max-width:720px;margin:auto;padding:24px;border:1px solid #ddd;border-radius:16px;box-shadow:0 6px 20px rgba(0,0,0,.06)}
      .row{display:flex;gap:12px;align-items:center}
      .btn{padding:10px 16px;border:0;border-radius:10px;background:#111;color:#fff;cursor:pointer}
      .muted{color:#666}
      img{max-width:100%;border:1px solid #eee;border-radius:12px}
      code{background:#f7f7f7;padding:2px 6px;border-radius:6px}
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Redline Agent</h1>
      <p class="muted">Upload a <code>.dxf</code>, then run review. AI can be toggled.</p>
      <div class="row">
        <input type="file" id="file" accept=".dxf" />
        <button class="btn" onclick="upload()">Upload</button>
        <label><input type="checkbox" id="staticRules" checked/> Static rules</label>
        <label><input type="checkbox" id="ai" checked/> AI</label>
        <button class="btn" onclick="review()">Review</button>
      </div>
      <pre id="log"></pre>
      <div id="preview"></div>
    </div>

    <script>
      let fileName = null;
      const log = (t) => document.getElementById('log').textContent = t;

      async function upload(){
        const f = document.getElementById('file').files[0];
        if(!f){ alert('Pick a .dxf first'); return; }
        const fd = new FormData();
        fd.append('file', f);
        const r = await fetch('/api/upload', { method: 'POST', body: fd });
        const j = await r.json();
        if(r.ok){ fileName = j.file_name; log('Uploaded: '+fileName); }
        else{ log('Error: '+(j.detail||JSON.stringify(j))); }
      }

      async function review(){
        if(!fileName){ alert('Upload first'); return; }
        const body = { file_name: fileName, ai: document.getElementById('ai').checked, static_rules: document.getElementById('staticRules').checked };
        const r = await fetch('/api/review', { method: 'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify(body) });
        const j = await r.json();
        if(!r.ok){ log('Error: '+(j.detail||JSON.stringify(j))); return; }
        log(JSON.stringify(j, null, 2));
        const box = document.getElementById('preview');
        box.innerHTML = '';
        if(j.preview_png){
          const img = document.createElement('img');
          img.src = j.preview_png; box.appendChild(img);
        }
        if(j.annotated_dxf){
          const a = document.createElement('a');
          a.textContent = 'Download annotated DXF';
          a.href = j.annotated_dxf; a.download = '';
          box.appendChild(a);
        }
        if(j.preview_pdf){
          const a2 = document.createElement('a');
          a2.textContent = 'Download PDF preview';
          a2.href = j.preview_pdf; a2.download = '';
          box.appendChild(document.createElement('br'));
          box.appendChild(a2);
        }
      }
    </script>
  </body>
</html>
```

---

## Dockerfile (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DATA_DIR=/app/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## README.md

````md
# Redline Agent

AI-assisted CAD redlining for DXF files.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env  # set AI_PROVIDER=openai|groq|none and API keys
uvicorn app.main:app --reload
````

Open [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)

## Notes

* MVP supports **DXF only**. Convert DWG → DXF using ODA File Converter or Autodesk cloud APIs.
* Previews use Matplotlib and may not render every CAD feature perfectly; for production fidelity, integrate Autodesk Forge/Platform services.
* Add your own static rules in `app/rules.py` and adjust AI prompts in `app/ai.py`.

```
```
