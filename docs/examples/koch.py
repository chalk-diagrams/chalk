# Based on the following example from Diagrams
# https://archives.haskell.org/projects.haskell.org/diagrams/gallery/Koch.html

from PIL import Image as PILImage
from chalk import *
from chalk.transform import *
from chalk.trail import unit_x
import chalk


def koch(n):
    if n == 0:
        return unit_x.scale_x(5)
    else:
        return (
            koch(n - 1).scale(1 / 3)
            + koch(n - 1).scale(1 / 3).rotate_by(-1 / 6)
            + koch(n - 1).scale(1 / 3).rotate_by(+1 / 6)
            + koch(n - 1).scale(1 / 3)
        )

d = vcat(koch(i).stroke().line_width(0.01) for i in range(1, 5))


# Render
height = 512
d.render_svg("examples/output/koch.svg", height)
d.render("examples/output/koch.png", height)
PILImage.open("examples/output/koch.png")
