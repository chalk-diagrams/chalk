from chalk import *
from chalk.trail import Trail
from colour import Color
from PIL import Image as PILImage

grey = Color("grey")
blue = Color("blue")
orange = Color("orange")

octagon = regular_polygon(8, 1.5).rotate_by(1 / 16).line_color(grey).line_width(0.5).show_origin()
dias = octagon.named("first") | hstrut(3) | octagon.named("second")

ex1 = dias.connect(
    "first",
    "second",
    ArrowOpts(trail=Trail.from_offsets([unit_x, 0.25 * unit_y, unit_x, 0.25 * unit_y]))
)
ex1

ex1 = dias.connect(
    "first",
    "second",
    ArrowOpts(
        head_style=Style.empty().fill_color(grey),
        arc_height=0.5,
        shaft_style=Style.empty().line_color(blue),
    ),
)
ex1 = ex1.connect(
    "second",
    "first",
    ArrowOpts(
        head_style=Style.empty().fill_color(grey),
        arc_height=0.5,
        shaft_style=Style.empty().line_color(blue),
    ),
)

ex12 = ex1.connect_perim(
    "first",
    "second",
    unit_x.rotate_by(15 / 16),
    unit_x.rotate_by(9 / 16),
    ArrowOpts(head_pad=0.1),
)
ex3 = arrow_v(unit_y)
d = ex12 + ex3

d

output_path = "examples/output/arrows.svg"
d.render_svg(output_path, height=200)

output_path = "examples/output/arrows.png"
d.render(output_path, height=200)

PILImage.open(output_path)

output_path = "examples/output/arrows.pdf"
d.render_pdf(output_path, height=200)
