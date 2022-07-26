from chalk import *
from chalk.trail import Trail, Vec2Array
from colour import Color
from PIL import Image as PILImage

grey = Color("grey")
blue = Color("blue")
orange = Color("orange")

c = circle(0.5).fill_color(orange).named("c")
c = (
    c.connect_perim("c", "c", unit_x, unit_y, ArrowOpts(arc_height=0.5))
    + trail.unit_x.scale(0.5).stroke()
    + trail.unit_y.scale(0.5).stroke()
)
c

arrowV(unit_y).render_pdf("test.pdf", 200)

octagon = (
    polygon(8, 1.0).rotate_by(1 / 16).line_color(grey).line_width(0.25).show_origin()
)
dias = octagon.named("first") | hstrut(3) | octagon.named("second")

ex1 = dias.connect(
    "first",
    "second",
    ArrowOpts(trail=Trail(Vec2Array([(1, 0), (0, 0.25), (1, 0), (0, -0.25)]))),
)
ex1

ex1 = dias.connect(
    "first",
    "second",
    ArrowOpts(
        head_style=Style(fill_color=grey),
        arc_height=0.5,
        shaft_style=Style(line_color=blue),
    ),
)
ex12 = ex1.connect_perim(
    "first",
    "second",
    unit_x.rotate_by(15 / 16),
    unit_x.rotate_by(9 / 16),
    ArrowOpts(head_pad=0.1),
)
ex3 = arrowV(unit_y)
d = ex12 + ex3

d

output_path = "examples/output/arrows.svg"
d.render_svg(output_path, height=200)

output_path = "examples/output/arrows.png"
d.render(output_path, height=200)

PILImage.open(output_path)

output_path = "examples/output/arrows.pdf"
d.render_pdf(output_path, height=200)
