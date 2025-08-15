import ezdxf
from ezdxf.addons.drawing import matplotlib as ezdxf_matplotlib
import matplotlib.pyplot as plt

def render_preview(dxf_path: str, png_path: str, pdf_path: str):
    # Load DXF
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Setup Matplotlib figure and backend
    fig = plt.figure(figsize=(8, 8))
    ctx = ezdxf_matplotlib.MatplotlibBackend()  # REMOVE dpi here

    # Draw entities
    ezdxf_matplotlib.draw_entities(msp, ctx)
    ax = ctx.ax
    ax.set_aspect('equal')
    ax.autoscale()

    # Save PNG
    plt.savefig(png_path, dpi=200, bbox_inches='tight')  # dpi here
    # Save PDF
    plt.savefig(pdf_path, bbox_inches='tight')  
    plt.close(fig)
