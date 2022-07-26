# Example taken from
# https://diagrams.github.io/doc/quickstart.html

from colour import Color
from chalk import *


white = Color("white")
green = Color("green")
pink = Color("pink")


def make_node(n):
    c = circle(0.2).fill_color(green)
    t = text(str(n), 0.2).fill_color(white).line_color(white).line_width(0)
    return c + t


n = 6
hexagon = Path.regular_polygon(n, 1)
nodes = [make_node(i).named(i) for i in range(n)]
nodes = [node.translate(point.x, point.y) for node, point in zip(nodes, hexagon.points)]
dia = concat(nodes) 

for i in range(n):
    for j in range(i + 1, n):
        dia = dia.connect_outside(i, j, ArrowOpts(head_pad=0.1, tail_pad=0.1))

dia.render_svg("examples/output/tournament-network.svg")
