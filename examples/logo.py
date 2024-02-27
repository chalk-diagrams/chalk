from colour import Color
from chalk import *

# This code is for a Vogel subflower, ported from:
# https://diagrams.github.io/gallery/Sunflower.html

black = Color("#000000")
white = Color("white")
grey = Color("#cccccc")

def coord(m):
    return from_polar(math.sqrt(m)/1.2, 2.4 * m)

def from_polar(r, theta):
    return (r * math.cos(theta), r * math.sin(theta))

def mkCoords(n):
    return [coord(i) for i in range(1, n+1)]

def floret(r):
    n = math.floor(1.8 * math.sqrt(r)) % 5
    # Hippie color palatte.
    colors = [Color(h) for h in ["#18b0dc",
                                 "#056753",
                                 "#b564ac",
                                 "#e0b566",
                                 "#e52828"]
    ]

    return circle(0.6).line_width(0).fill_color(colors[n])

def florets(m):
    return [floret(math.sqrt(i)) for i in range(1,m+1)]
    
def sunflower(n):
    return concat(flor.translate(cord[0], cord[1]) for cord, flor in zip(mkCoords(n), florets(n)))
        
floret = sunflower(1900).center_xy().scale_uniform_to_x(1).center_xy()
background = rectangle(1.6, 1).fill_color(black).line_width(0).translate(-0.15, 0)
logo = text("Chalk", 0.35).fill_color(grey).line_width(0.1).line_color(black).translate(-0.4, -0.1)
mask = rectangle(1.6, 0.6).translate(-0.15, 0)

# assemble
d = (background + floret + logo).align_t().with_envelope(mask.align_t())

d.render("examples/output/logo.png", 500)
d.render_svg("examples/output/logo.svg", 500)

d.render_pdf("examples/output/logo.pdf", 50)
