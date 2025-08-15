import os
import json
from typing import List
from dotenv import load_dotenv
from .models import Entity, Issue

load_dotenv()

# ⚡ AI is hardwired to Groq + llama3-70b-8192
AI_PROVIDER = "groq"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

try:
    from groq import Groq as _Groq
except Exception:
    _Groq = None


def summarize_entities_for_llm(entities: List[Entity], limit: int = 300) -> str:
    """Compact summary to keep token usage manageable."""
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
    if not GROQ_API_KEY or not _Groq:
        # No Groq API key or package available
        return []

    summary = summarize_entities_for_llm(entities)

    system_prompt = (
        "You are a senior CAD reviewer. Given a structured list of entities from a DXF, "
        "find likely drafting issues. Prefer high-precision items (labels, obvious dimensions). "
        "Return ONLY a JSON array of objects with keys x, y, issue, meta."
    )

    user_prompt = (
        f"Entities (truncated):\n{summary}\n\n"
        "Rules of thumb: rooms must have labels; doors ≥ 800mm clear; no text on layer '0'."
    )

    # Initialize Groq client
    client = _Groq(api_key=GROQ_API_KEY)

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    # Extract content (handle Groq dict response)
    content = (
        resp.choices[0].message["content"]
        if isinstance(resp.choices[0].message, dict)
        else resp.choices[0].message.content
    )

    try:
        raw = json.loads(content)
        issues: List[Issue] = []
        for it in raw:
            issues.append(
                Issue(
                    x=float(it.get("x", 0.0)),
                    y=float(it.get("y", 0.0)),
                    issue=str(it.get("issue", "AI-flagged")),
                    meta=it.get("meta", {}),
                )
            )
        return issues
    except Exception:
        # If LLM didn't return pure JSON, ignore AI output gracefully
        return []
