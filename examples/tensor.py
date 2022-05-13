from chalk import *
from colour import Color
import chalk.transform as tx

def hstrut(width=0.2): return hrule(width).line_width(0)
h = hstrut(2.5)

papaya = Color("#ff9700")
white = Color("white")
black = Color("black")

# Text
def label(te, s=1.5):
    return (text(te, s).fill_color(black).line_color(white).pad_t(2.5).center_xy())

def tensor(depth, rows, columns):
    "Draw a tensor"
    
    up = Vector(0, -1)
    right = Vector(1, 0)
    hyp = (up * 0.5).apply_transform(tx.ShearX(-1))
    
    # Faces
    face_m = rectangle(1, 1).align_tl()
    face_t = rectangle(1, 0.5).shear_x(-1).align_bl()
    face_r = rectangle(0.5, 1).shear_y(-1).align_tr()
    

    # Make a single cube with bounding box around the front face
    cube = (face_m + face_t).align_t().align_r() + face_r
    return concat(([cube.translate_by(hyp * i + -up * j + right * k)
                    for i in reversed(range(depth))
                    for j in reversed(range(rows))
                    for k in range(columns)]))

def t(d, r, c):
    return tensor(d, r, c).fill_color(white)



d, r, c = 3, 4, 5

base = t(d, r, c).line_color(papaya)
m = hcat([t(1, r, c),  t(d, 1, c), label("→"), (base + t(1, r, c)), (base + t(d, 1, c) ), label("="), t(d, r, c)], sep=2.5)


pathsvg = "examples/output/tensor.svg"
m.line_width(0.02).render_svg(pathsvg, 500)
path = "examples/output/tensor.png"
m.line_width(0.02).render(path, 500)


