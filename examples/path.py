from chalk import *
from chalk.segment import Segment
from chalk.arc import ArcSegment
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

d0 = circle(1).show_origin()

d1 = Path([ArcSegment.arc_between(P2(-1, 0), P2(1, 0.0), 1)]).stroke().show_origin()

d2 = Path([ArcSegment.arc_between(P2(-1, 0), P2(1, 0.0), -1)]).stroke().show_origin()

d3 = Path([ArcSegment.arc_between(P2(-1, 0), P2(0, 1), 1)]).stroke().show_origin()

d4 = Path([ArcSegment.arc_between(P2(-1, 0), P2(0, 1), -1)]).stroke().show_origin()

d5 = Path([ArcSegment.arc_between(P2(-1, 0), P2(1, 0.0), 1).scale_y(0.5)]).stroke().show_origin()

d5 = Path([ArcSegment.arc_between(P2(-1, 0), P2(1, 0.0), 1).scale_y(0.5).rotate(45)]).stroke().show_origin()

# d = Path([Segment(P2(0, r), P2(0, b)),
#           ArcSegment.arc_between(P2(0, b), P2(r, 1), rad)]).stroke()

d6 = Path([Segment(P2(0, r), P2(0, b)),
          ArcSegment.arc_between(P2(0, b), P2(r, 1), -rad),
           Segment(P2(r, 1), P2(b, 1)),
           ArcSegment.arc_between(P2(b, 1), P2(1, 1-r), -rad),
          Segment(P2(1, 1-r), P2(1, r)),
          ArcSegment.arc_between(P2(1, r), P2(1-r, 0), -rad),
          Segment(P2(1-r, 0), P2(r, 0)),
          ArcSegment.arc_between(P2(r, 0), P2(0, r), -rad)

]).stroke().center_xy().show_origin()

d = d0 / vstrut(1) / d1 / vstrut(1) / d2 / vstrut(1) / d3 / vstrut(1) / d4  / vstrut(1) / d5 / vstrut(1) / d6
# d = d6
# d = d1
d = d.fill_color(Color("blue"))

d.render_svg("examples/output/path.svg", height=300)
d.render_pdf("examples/output/path.pdf", height=300)
d.render("examples/output/path.png", height=300)
