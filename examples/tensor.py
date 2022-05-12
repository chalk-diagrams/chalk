from chalk import *
from colour import Color

def hstrut(width=0.2): return hrule(width).line_width(0)
h = hstrut(2.5)

papaya = Color("#ff9700")
white = Color("white")
black = Color("black")

# Text
def label(te, s=1.5):
    return (text(te, s).fill_color(black).line_color(white).pad_t(2.5).center_xy())

def tensor(skew, depth, rows, columns):
    "Draw a tensor"
    s = Vector(0, 0)
    up = Vector(0, -1)
    right = Vector(1, 0)
    hyp = Vector(skew, -math.sqrt(1 - skew))
    def quad(v1, v2):
        return Trail([s, v1, v2, -v1, -v2]).stroke()
    # Faces
    face_t, face_r, face_m = quad(-hyp, right), quad(up, hyp), quad(-up, right) 

    # Make a single cube with bounding box around the front face
    cube = (face_m + face_t.align_l().align_b()).align_t().align_r() + face_r.align_t().align_r()
    return concat(([cube.translate_by(hyp * i + -up * j + right * k)
                    for i in reversed(range(depth))
                    for j in reversed(range(rows))
                    for k in range(columns)]))

def t(d, r, c):
    return tensor(0.4, d, r, c)


m = tensor(0.4, 5, 8, 3) | h | tensor(0.6, 5, 8, 3,)

d, r, c = 3, 4, 5

base = t(d, r, c).line_color(papaya)
m = t(1, r, c) | h |  t(d, 1, c) | h | label("→") | h |  (base + t(1, r, c)) | h | (base + t(d, 1, c) )  | h | label("=") | h | t(d, r, c)


pathsvg = "examples/output/tensor.svg"
m.line_width(0.02).render_svg(pathsvg, 500)
path = "examples/output/tensor.png"
m.line_width(0.02).render(path, 500)


