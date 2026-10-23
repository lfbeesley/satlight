"""Draw the pose/update/render flow chart for guide/articulation.md.

Writes pose_flow.svg (HTML, on the site's dark pink palette) and pose_flow.png
(PDF, black on white) next to this script. Sphinx's `pose_flow.*` picks the SVG
for HTML and the PNG for LaTeX. Re-run after editing; the outputs are
committed, not built on Read the Docs.
"""

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

mpl.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 10,
    "svg.fonttype": "path",        # glyphs as outlines: no reliance on viewer fonts
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

# Pink Cat Boo, matching html_theme_options in conf.py
DARK = dict(bg="#202330", box="#2d2f42", shared="#472541", ink="#FFF0F5",
            edge="#fe7c8e", note="#C5C8C6", code="#f3b4bf", arrow="#fe7c8e")
PRINT = dict(bg="white", box="white", shared="0.94", ink="black",
             edge="black", note="0.45", code="black", arrow="black")


def draw(c, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    fig.patch.set_facecolor(c["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    mono = dict(family="DejaVu Sans Mono", size=8, color=c["code"])

    def box(x, y, w, h, title, code=None, note=None, fill=c["box"]):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fill, ec=c["edge"], lw=0.8))
        lines = [(title, dict(color=c["ink"]))]
        if code:
            lines.append((code, mono))
        if note:
            lines.append((note, dict(color=c["note"], size=9, style="italic")))
        step = 0.42
        y0 = y + step * (len(lines) - 1) / 2
        for i, (text, kw) in enumerate(lines):
            ax.text(x, y0 - i * step, text, ha="center", va="center", **kw)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=10, lw=0.9, color=c["arrow"]))

    # Column headings
    ax.text(2.45, 6.65, "you, through bpy", ha="center", color=c["note"], style="italic")
    ax.text(7.55, 6.65, "satlight", ha="center", color=c["note"], style="italic")

    box(2.45, 5.75, 4.3, 1.25, "where the parts are",
        "o.matrix_world = ...", "moves components")
    box(7.55, 5.75, 4.3, 1.25, "Sun, observer and camera",
        "renderer.update(geom)", "never touches the parts")
    box(5.0, 3.55, 5.2, 1.05, "Blender scene (shared state)",
        note="parts and camera as they are now", fill=c["shared"])
    box(5.0, 1.35, 5.2, 1.25, "take the picture", "renderer.render()",
        note="casts rays at the scene as it is now")
    ax.text(5.0, 0.35, "→  flux at the observer (W m$^{-2}$)",
            ha="center", va="center", color=c["ink"])

    arrow(2.9, 5.1, 4.1, 4.1)
    arrow(7.1, 5.1, 5.9, 4.1)
    arrow(5.0, 3.0, 5.0, 2.0)

    fig.savefig(path, dpi=200, facecolor=c["bg"])
    plt.close(fig)


here = pathlib.Path(__file__).resolve().parent
draw(DARK, here / "pose_flow.svg")
draw(PRINT, here / "pose_flow.png")
