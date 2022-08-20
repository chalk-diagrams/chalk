from PIL import Image as PILImage
from chalk import *
from colour import Color

h = hstrut(2.5)
papaya = Color("#ff9700")
white = Color("white")
black = Color("black")


def draw_cube():
    # Assemble cube
    face_m = rectangle(1, 1).align_tl()
    face_t = rectangle(1, 0.5).shear_x(-1).align_bl()
    face_r = rectangle(0.5, 1).shear_y(-1).align_tr()
    cube = (face_t + face_m).align_tr() + face_r

    # Replace envelope with front face. 
    return cube.align_bl().with_envelope(face_m.align_bl())

def draw_tensor(depth, rows, columns):
    "Draw a tensor"
    cube  = draw_cube()
    # Fix this ...
    hyp = (unit_y * 0.5).shear_x(-1)
    # Build a matrix. 
    front = cat([hcat([cube for i in range(columns)])
                 for j in reversed(range(rows))], -unit_y).align_t()

    # Build depth
    return concat(front.translate(-k * hyp.x, -k * hyp.y)
                  for k in reversed(range(depth)))

draw_tensor(2, 3, 4)

def t(d, r, c):
    return draw_tensor(d, r, c).fill_color(white)

def label(te, s=1.5):
    return (text(te, s).fill_color(black).line_color(white).center_xy())


# Create a diagram.
d, r, c = 3, 4, 5
base = t(d, r, c).line_color(papaya)
m = hcat([t(1, r, c),  t(d, 1, c), label("→"), (base + t(1, r, c)), (base + t(d, 1, c) ), label("="), t(d, r, c)], sep=2.5).line_width(0.02)
m
