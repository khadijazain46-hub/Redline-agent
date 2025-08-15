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