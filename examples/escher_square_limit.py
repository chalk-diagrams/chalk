# This example is based on a corresponding example from Lisp by Frank Buss:
# https://frank-buss.de/lisp/functional.html
# A more general implementation is provided by Jeremy Gibbons using diagrams in Haskell:
# https://archives.haskell.org/projects.haskell.org/diagrams/gallery/SquareLimit.html

import math

from toolz import take, iterate  # type: ignore

from chalk import concat, make_path, square, strut


# fmt: off
markings = {
    "p": [
        [(4, 4), (6, 0)],
        [(0, 3), (3, 4), (0, 8), (0, 3)],
        [(4, 5), (7, 6), (4, 10), (4, 5)],
        [(11, 0), (10, 4), (8, 8), (4, 13), (0, 16)],
        [(11, 0), (14, 2), (16, 2)],
        [(10, 4), (13, 5), (16, 4)],
        [(9, 6), (12, 7), (16, 6)],
        [(8, 8), (12, 9), (16, 8)],
        [(8, 12), (16, 10)],
        [(0, 16), (6, 15), (8, 16), (12, 12), (16, 12)],
        [(10, 16), (12, 14), (16, 13)],
        [(12, 16), (13, 15), (16, 14)],
        [(14, 16), (16, 15)],
    ],
    "q": [
        [(2, 0), (4, 5), (4, 7)],
        [(4, 0), (6, 5), (6, 7)],
        [(6, 0), (8, 5), (8, 8)],
        [(8, 0), (10, 6), (10, 9)],
        [(10, 0), (14, 11)],
        [(12, 0), (13, 4), (16, 8), (15, 10), (16, 16), (12, 10), (6, 7), (4, 7), (0, 8)],
        [(13, 0), (16, 6)],
        [(14, 0), (16, 4)],
        [(15, 0), (16, 2)],
        [(0, 10), (7, 11)],
        [(9, 12), (10, 10), (12, 12), (9, 12)],
        [(8, 15), (9, 13), (11, 15), (8, 15)],
        [(0, 12), (3, 13), (7, 15), (8, 16)],
        [(2, 16), (3, 13)],
        [(4, 16), (5, 14)],
        [(6, 16), (7, 15)],
    ],
    "r": [
        [(0, 12), (1, 14)],
        [(0, 8), (2, 12)],
        [(0, 4), (5, 10)],
        [(0, 0), (8, 8)],
        [(1, 1), (4, 0)],
        [(2, 2), (8, 0)],
        [(3, 3), (8, 2), (12, 0)],
        [(5, 5), (12, 3), (16, 0)],
        [(0, 16), (2, 12), (8, 8), (14, 6), (16, 4)],
        [(6, 16), (11, 10), (16, 6)],
        [(11, 16), (12, 12), (16, 8)],
        [(12, 12), (16, 16)],
        [(13, 13), (16, 10)],
        [(14, 14), (16, 12)],
        [(15, 15), (16, 14)],
    ],
    "s": [
        [(0, 0), (4, 2), (8, 2), (16, 0)],
        [(0, 4), (2, 1)],
        [(0, 6), (7, 4)],
        [(0, 8), (8, 6)],
        [(0, 10), (7, 8)],
        [(0, 12), (7, 10)],
        [(0, 14), (7, 13)],
        [(8, 16), (7, 13), (7, 8), (8, 6), (10, 4), (16, 0)],
        [(10, 16), (11, 10)],
        [(10, 6), (12, 4), (12, 7), (10, 6)],
        [(13, 7), (15, 5), (15, 8), (13, 7)],
        [(12, 16), (13, 13), (15, 9), (16, 8)],
        [(13, 13), (16, 14)],
        [(14, 11), (16, 12)],
        [(15, 9), (16, 10)],
    ],
}
# fmt: on

names = "pqrs"
blank = strut(1, 1)
θ = 90


def normalize(coords):
    def center(val: float) -> float:
        return (val - 8) / 16

    return [(center(x), -center(y)) for x, y in coords]


def make_tile(name):
    return concat(make_path(normalize(coords)) for coords in markings[name])

def quartet(tl, tr, bl, br):
    diagram = (tl | tr) / (bl | br)
    return diagram.center_xy().scale(0.5)


def cycle(diagram):
    tl = diagram
    tr = diagram.rotate(θ).rotate(θ).rotate(θ)
    bl = diagram.rotate(θ)
    br = diagram.rotate(θ).rotate(θ)
    return quartet(tl, tr, bl, br)

fish = {name: make_tile(name) for name in names}
fish_t = quartet(fish["p"], fish["q"], fish["r"], fish["s"])
fish_u = cycle(fish["q"].rotate(θ))

side_1 = quartet(blank, blank, fish_t.rotate(θ), fish_t)

side_2 = quartet(side_1, side_1, fish_t.rotate(θ), fish_t)
corner_1 = quartet(blank, blank, blank, fish_u)

corner_2 = quartet(corner_1, side_1, side_1.rotate(θ), fish_u)
pseudocorner = quartet(corner_2, side_2, side_2.rotate(θ), fish_t.rotate(θ))


pseudolimit = cycle(pseudocorner).line_width(0.05)

output_path = "examples/output/escher-square-limit.png"
pseudolimit.render(output_path, height=512)

# SVG render
output_path = "examples/output/escher-square-limit.svg"
pseudolimit.render_svg(output_path, height=512)


# output_path = "examples/output/escher-square-limit.pdf"
# pseudolimit.render_pdf(output_path, height=512)
