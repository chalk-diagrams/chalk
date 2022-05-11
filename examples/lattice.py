from chalk import *
from colour import Color

black = Color("#000000")
white = Color("#ffffff")
STEPS = 7
NODES = 7
x = hcat(vcat((rectangle(1, 0.4, 0.1).named((i, j)) + text(f"Node {i} {j}", 0.2).fill_color(black)) / vrule(1).line_width(0) for j in range(NODES)) | hrule(2).line_width(0)
         for i in range(STEPS))

connects = []
for i in range(NODES-1):
    for j in range(STEPS):
        for j2 in range(STEPS):
            connects.append(connect_outer(x, (i, j), "E", (i+1, j2), "W").line_color(black))
d = x + concat(connects)


path = "examples/output/lattice.svg"
d.render_svg(path, height=256)

path = "examples/output/lattice.png"
d.render(path, height=256)
