from chalk import *
from chalk.transform import *


# Draw a space filling Hilbert curve

def hilbert(n):
    def hilbert2(m): return hilbert(m).rotate(-math.pi / 2)
    if n == 0: return Trail([])
    h, h2 = hilbert(n -1), hilbert2(n-1)
    return (h2.reflect_y() + Trail.from_path(vrule(1))
            + h + Trail.from_path(hrule(1))
            + h + Trail.from_path(vrule(-1))
            + h2.reflect_x())

d = hilbert(5).stroke().line_width(0.05)
d.render_svg("examples/output/hilbert.svg", 500)
d.render("examples/output/hilbert.png", 500)
          
