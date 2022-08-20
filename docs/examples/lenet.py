from PIL import Image as PILImage
from chalk import *
from colour import Color
from chalk import BoundingBox

# Colors
papaya = Color("#ff9700")
blue = Color("#005FDB")
black = Color("#000000")
white = Color("#ffffff")
grey = Color("#bbbbbb")


# Some general functions

def label(te):
    "Create text."
    return text(te, 2).fill_color(black).line_width(0)

def cover(d, a, b):
    "Draw a bounding_box around a subdiagram"
    b1 = d.get_subdiagram_envelope(a)
    b2 = d.get_subdiagram_envelope(b)
    new_bb = b1 + b2
    return rectangle(new_bb.width, new_bb.height) \
            .translate(new_bb.center.x, new_bb.center.y)

def tile(d, m, n, name = ""):
    "Tile a digram with names"
    return hcat(vcat(d.named((name, j, i)) for j in range(n)) for i in range(m)).center_xy()

def connect_all(d, a, b):
    "Connect all corners of two diagrams"

    for x_border in [-unit_x, unit_x]:
        for y_border in [-unit_y, unit_y]:
            p = x_border + y_border
            d = d.connect_perim(a, b, p, p, ArrowOpts(head_arrow=empty()))
    return d

# NN drawing
def cell():
    return rectangle(1, 1).line_width(0.01)

def matrix(n, r, c):
    return tile(cell(), c, r, n)

def back(r, n):
    "Backing stack"
    return concat((r.translate(-i/2, -i/2).fill_opacity((n - i + n /2) / n)
                   for i in range(n-1, -1, -1)))

lw = 0.05
def stack(n, size, l, top, bot):
    "Feature map stack"
    m = matrix(n, size, size).fill_color(Color("#dddddd"))
    r = rectangle(size, size).fill_color(grey).line_width(lw)
    return (label(top) / (back(r, l) + m) / label(bot)).center_xy()
stack("a", 32, 0, "", "")

def network(n, size, top, bot):
    "Draw a network layer"
    return (label(top) /  rectangle(2, size).fill_color(grey).line_width(lw).named(n) / label(bot)).center_xy()

# The number 7
draw = make_path([(-10, -10), (10, -10 ), (-10, 10)]).line_width(0.09).line_color(blue).fill_opacity(0)

# Draw the main diagram.
h = hstrut(6.5)
d = ((stack("a", 32, 0, "", "") + draw) | (label("conv") / h) |
     stack("b", 28, 6, "", "C1") | (label("pool") / h) |
     stack("c", 14, 6, "", "S2") | (label("conv") / h) |
     stack("d", 10, 16, "", "C3") | (label("pool") / h) | hstrut(-0.5) |
     stack("e", 5, 16, "", "S4") | (label("dense") / h) |
     network("dense1",  12, "", "") | (label("dense") / (h)) |
     network("dense2",  8.4, "", "") | (label("dense") / h) |
     network("dense3",  1, "", ""))

d = d.scale_uniform_to_x(5)

# Draw the orange boxes
boxes = [(("a", 2, 2), ("a", 6, 6)),
         (("b", 2, 2), ("b", 2, 2)),
         (("b", 20, 2), ("b", 23, 5)),
         (("c", 10, 2), ("c", 11, 3)),
         (("c", 4, 6), ("c", 8, 10)),
         (("d", 4, 6), ("d", 4, 6)),
         (("d", 6, 4), ("d", 9, 7)),
         (("e", 3, 2), ("e", 4, 3))]

d += concat([cover(d, *b).fill_color(papaya).fill_opacity(0.3).named(("box", i))
             for i, b in enumerate(boxes)])

connect = [(("box", i), ("box", i + 1)) for i in range(0, 6, 2)]


for b in connect:
    d = connect_all(d, *b)

set_svg_height(300)
d
