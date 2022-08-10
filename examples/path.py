from chalk import *
from chalk.shapes.segment import Segment
from chalk.shapes.arc import ArcSegment
from colour import Color
import math
# d = Path([ArcSegment.arc_between(P2(0, 0), P2(2, 0), 1), ArcSegment.arc_between(P2(2, 0), P2(0, 0), 1)]).stroke()
# d = Primitive.from_shape(Path([Segment(P2(0, 0), P2(1, 1))]))
# d = Primitive.from_shape(Path([Segment(P2(0, 0), P2(1, 1)),
#                                Segment(P2(1, 1), P2(1, 2))]))

# d = Path([ArcSegment.arc_between(P2(0, 0), P2(2, 0), 1)]).stroke()
r = 0.2
b = 1 - r
rad = 0.2 / 4

a  = ArcSegment(10, 50).rotate(90)

a  = ArcSegment.arc_between(P2(-1, 0), P2(0, 1.0), 1)
# print(a.p, a.q, a.r_x, a.r_y, a.angle, a.tangle)
d = []

d += [circle(1).show_origin()]

d += [Path([ArcSegment.arc_between(P2(-1, 0), P2(1, 0.0), 1)]).stroke().show_origin()]

d += [Path([ArcSegment.arc_between(P2(-1, 0), P2(1, 0.0), -1)]).stroke().show_origin()]

d += [Path([ArcSegment.arc_between(P2(-1, 0), P2(0, 1), 1)]).stroke().show_origin()]

d += [Path([ArcSegment.arc_between(P2(-1, 0), P2(0, 1), -1)]).stroke().show_origin()]

d += [Path([ArcSegment.arc_between(P2(-1, 0), P2(1, 0.0), 1).scale_y(0.5).rotate(45)]).stroke().show_origin()]

# d = Path([Segment(P2(0, r), P2(0, b)),
#           ArcSegment.arc_between(P2(0, b), P2(r, 1), rad)]).stroke()

d += [Path([Segment(P2(0, b/2), P2(0, b)),
          ArcSegment.arc_between(P2(0, b), P2(r, 1), -rad),
           Segment(P2(r, 1), P2(b, 1)),
           ArcSegment.arc_between(P2(b, 1), P2(1, 1-r), -rad),
          Segment(P2(1, 1-r), P2(1, r)),
          ArcSegment.arc_between(P2(1, r), P2(1-r, 0), -rad),
          Segment(P2(1-r, 0), P2(r, 0)),
           ArcSegment.arc_between(P2(r, 0), P2(0, r), -rad),
           Segment(P2(0, r), P2(0, b / 2))

]).stroke().center_xy().show_origin()]


d += [cat(
    [
        ArcSegment(180, 135).stroke().show_origin(),
        ArcSegment(180, 135).scale_x(-1).stroke().show_origin(),
        ArcSegment(180, 135).stroke().scale_x(-1).show_origin(),
    ],
    unit_x,
    sep=0.5
)]

d = vcat(d, sep=1.0)

d = d.fill_color(Color("blue"))

d.render_svg("examples/output/path.svg", height=300)
d.render_pdf("examples/output/path.pdf", height=300)
d.render("examples/output/path.png", height=300)
