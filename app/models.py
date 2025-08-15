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

