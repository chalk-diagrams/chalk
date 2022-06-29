from PIL import Image as PILImage
from chalk import *
from colour import Color
import chalk.transform as tx
import math
h = hstrut(2.5)
papaya = Color("#ff9700")
white = Color("white")
black = Color("black")


def draw_cube():
    up = Vec2(0, -1)
    hyp = tx.Affine.shear(-tx.from_radians(math.atan(1)), 0) * (up * 0.5)
    right = Vec2(1, 0)
    
    # Faces
    face_m = rectangle(1, 1).align_tl()
    face_t = rectangle(1, 0.5).shear_x(-1).align_bl()
    face_r = rectangle(0.5, 1).shear_y(-1).align_tr()
    
    return (face_m + face_t).align_tr() + face_r, (up, hyp, right)
draw_cube()[0]
draw_cube()[0].render("examples/output/cube.png", 50)
draw_cube()[0].render_pdf("examples/output/cube.pdf", 50)

def draw_tensor(depth, rows, columns):
    "Draw a tensor"
    cube, (up, hyp, right) = draw_cube()
    return concat(([cube.translate_by(hyp * i + -up * j + right * k)
                    for i in reversed(range(depth))
                    for j in reversed(range(rows))
                    for k in range(columns)]))
draw_tensor(2, 3, 4)

def t(d, r, c):
    return draw_tensor(d, r, c).fill_color(white)

def label(te, s=1.5):
    return (text(te, s).fill_color(black).line_color(white).pad_t(2.5).center_xy())


# Create a diagram.
d, r, c = 3, 4, 5
base = t(d, r, c).line_color(papaya)
m = hcat([t(1, r, c),  t(d, 1, c), label("→"), (base + t(1, r, c)), (base + t(d, 1, c) ), label("="), t(d, r, c)], sep=2.5).line_width(0.02)


pathsvg = "examples/output/tensor.svg"
m.render_svg(pathsvg, 500)
path = "examples/output/tensor.png"
m.render(path, 500)
PILImage.open(path)

# Needed for recursion 
import sys
sys.setrecursionlimit(10000)

m.render_pdf("examples/output/tensor.pdf", 50)
