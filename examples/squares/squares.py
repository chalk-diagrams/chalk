from PIL import Image as PILImage
import math
import random

from itertools import product

from colour import Color
from chalk import square, concat

random.seed(0)

def make_square():
    colors = [
        Color("#ff9700"),  # papaya
        Color("#005FDB"),  # blue
    ]

    # generate uniformly a value in [-max_angle, max_angle]
    max_angle = math.pi / 24.0
    θ = 2 * max_angle * random.random() - max_angle

    # pick a random color
    i = random.random() > 0.75
    color = colors[i]

    return square(0.75).line_color(color).rotate(θ)
make_square()

def make_group(num_squares=4):
    return concat(make_square() for _ in range(num_squares))
make_group()

disps = range(4)
diagram = concat(make_group().translate(x, y) for x, y in product(disps, disps))
diagram = diagram.line_width(0.02)

path = "examples/output/squares.svg"
diagram.render_svg(path, height=256)
path = "examples/output/squares.png"
diagram.render(path, height=256)
PILImage.open(path)
