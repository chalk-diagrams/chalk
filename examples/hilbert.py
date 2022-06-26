# Based on the following example from Diagrams
# https://archives.haskell.org/projects.haskell.org/diagrams/gallery/Hilbert.html

from PIL import Image as PILImage
from chalk import *
from chalk.transform import *
from chalk.trail import unit_x, unit_y


# Draw a space filling Hilbert curve

def hilbert(n):
    def hilbert2(m): return hilbert(m).rotate(-math.pi / 2)
    if n == 0: return Trail([])
    h, h2 = hilbert(n -1), hilbert2(n-1)
    return (h2.reflect_y() + unit_y
            + h + unit_x
            + h + unit_y.reflect_y()
            + h2.reflect_x())

d = hilbert(5).stroke().line_width(0.05)
d.render_svg("examples/output/hilbert.svg", 500)
d.render_pdf("examples/output/hilbert.pdf", 100)
d.render("examples/output/hilbert.png", 500)
PILImage.open("examples/output/hilbert.png")
