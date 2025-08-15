import ezdxf
from typing import List
from .models import Issue

def ensure_redline_layer(doc) -> None:
    """Ensure a REDLINES layer exists in the DXF."""
    if "REDLINES" not in doc.layers:
        doc.layers.add("REDLINES", color=1)  # 1 = red

def annotate_issues(dxf_path_in: str, dxf_path_out: str, issues: List[Issue]) -> None:
    """Add circles and text annotations for issues in DXF."""
    doc = ezdxf.readfile(dxf_path_in)
    msp = doc.modelspace()
    ensure_redline_layer(doc)

    for it in issues:
        x, y = float(it.x), float(it.y)
        # Circle for issue location
        msp.add_circle((x, y), radius=200.0, dxfattribs={"layer": "REDLINES"})
        # Text for issue description
        msp.add_text(it.issue, dxfattribs={"insert": (x + 250.0, y + 250.0), "layer": "REDLINES"})

    doc.saveas(dxf_path_out)
