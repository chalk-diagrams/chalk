# Example taken from
# https://diagrams.github.io/doc/quickstart.html

from colour import Color
from chalk import *


white = Color("white")
green = Color("green")
pink = Color("pink")


def make_node(n):
    c = circle(0.2).fill_color(green)
    t = text(str(n), 0.2).fill_color(white).line_color(white)
    return c + t


def connect_diagrams(d1, d2, gap=0.1):
    # TODO Make this functino work on named subdiagrams
    c1 = d1.get_bounding_box().center
    c2 = d2.get_bounding_box().center

    v = (c2 - c1)
    midpoint = c1 + 0.5 * v
    vs = d1.get_trace().trace_v(midpoint, -v)
    ve = d2.get_trace().trace_v(midpoint, +v)
    if not vs:
        s = midpoint
    else:
        s = midpoint + (1 - gap) * vs

    if not ve:
        e = midpoint
    else:
        e = midpoint + (1 - gap) * ve

    return Primitive.from_shape(Path([s, e], True))


n = 6
hexagon = Path.regular_polygon(n, 1)
nodes = [make_node(i) for i in range(n)]
nodes = [node.translate(point.x, point.y) for node, point in zip(nodes, hexagon.points)]

connections = concat(connect_diagrams(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n))
dia = concat(nodes) + connections

dia.render_svg("examples/output/tournament-network.svg")
