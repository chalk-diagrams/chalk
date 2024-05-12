import math
import random

from itertools import product

from chalk import *

random.seed(1337)


h = math.sqrt(3) / 2
h1 = math.cos(math.pi / 3)


def hexagon_tile():
    arc1 = arc(0.5, -from_radians(math.pi / 3), from_radians(math.pi / 3))
    return (
        arc1.translate(-1, 0)
        + vrule(2 * h)
        + arc1.rotate_by(1 / 2).translate(1, 0)
        # + polygon(6, 1)
    )

def rotated_hexagon_tile(n):
    return hexagon_tile().rotate_rad(-n * 2 * math.pi / 3)

def center_position(x, y):
    if x % 2 == 0:
        return (2 - h1) * x, 2 * y * h
    else:
        return (2 - h1) * x, (2 * y - 1) * h


def hex_variation(num_tiles):
    rows = list(range(num_tiles))
    cols = list(range(num_tiles))
    get_angle = lambda: random.randint(0, 2)
    diagrams = [rotated_hexagon_tile(get_angle()) for _ in product(rows, cols)]
    grid = [center_position(x, y) for x, y in product(rows, cols)]
    return place_at(diagrams, grid)


dia = hex_variation(12).line_width(0.05)
dia = dia.rotate_by(-1 / 4)

dia.render_svg("examples/output/hex-variation.svg", height=512)
try:
    dia.render("examples/output/hex-variation.png", height=512)
    dia.render_pdf("examples/output/hex-variation.pdf", height=512)
except ModuleNotFoundError:
    print("Need to install Cairo")

