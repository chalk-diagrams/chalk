import random
from colour import Color
from chalk import *
blue = Color("#005FDB")

path = "examples/output/rectangle-parade.png"
d = hcat(rectangle(1 + 5*random.random(), 1 + 5*random.random()).rotate_by(random.random())
         for i in range(1, 50)).fill_color(blue)
d.render(path, height=64)
