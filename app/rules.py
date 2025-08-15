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
