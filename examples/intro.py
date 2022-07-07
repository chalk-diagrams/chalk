from colour import Color
from chalk import *

# define some colors
papaya = Color("#ff9700")
blue = Color("#005FDB")


path = "examples/output/intro-01.png"
d = circle(1).fill_color(papaya)
d.render(path, height=64)


# # Alternative, render as svg
path = "examples/output/intro-01.svg"
d.render_svg(path, height=64)

# Alternative, render as pdf
path = "examples/output/intro-01.pdf"
d.render_pdf(path, height=64)


path = "examples/output/intro-02.png"
d = circle(0.5).fill_color(papaya) | square(1).fill_color(blue)
d.render(path, height=64)

path = "examples/output/intro-02.svg"
d.render_svg(path, height=64)

path = "examples/output/intro-02.pdf"
d.render_pdf(path)

path = "examples/output/intro-03.png"
d = hcat(circle(0.1 * i) for i in range(1, 6)).fill_color(blue)
d.render(path, height=64)

# Alternative, render as svg
path = "examples/output/intro-03.svg"
d.render_svg(path, height=64)

# Alternative, render as pdf
path = "examples/output/intro-03.pdf"
d.render_pdf(path)

path = "examples/output/intro-04.png"

def sierpinski(n: int, size: int) -> Diagram:
    if n <= 1:
        return triangle(size)
    else:
        smaller = sierpinski(n - 1, size / 2)
        return smaller.above((smaller | smaller).center_xy())

d = sierpinski(5, 4).fill_color(papaya)
d.render(path, height=256)

path = "examples/output/intro-04.svg"
d.render_svg(path, height=256)

path = "examples/output/intro-04.pdf"
d.render_pdf(path, height=256)
