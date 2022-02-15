import streamlit as st  # type: ignore
from colour import Color
from diagrams import *

# define some colors
papaya = Color("#ff9700")
blue = Color("#005FDB")

path = "examples/output/intro-01.png"
d = circle(1).fill_color(papaya)
d.render(path, height=64)
st.image(path)

path = "examples/output/intro-02.png"
d = circle(0.5).fill_color(papaya) | square(1).fill_color(blue)
d.render(path, height=64)
st.image(path)

path = "examples/output/intro-03.png"
d = hcat(circle(0.1 * i) for i in range(1, 6)).fill_color(blue)
d.render(path, height=64)
st.image(path)

path = "examples/output/intro-04.png"

def sierpinski(n: int, size: int) -> Diagram:
    if n <= 1:
        return triangle(size)
    else:
        smaller = sierpinski(n - 1, size / 2)
        return smaller.above(smaller.beside(smaller).center_xy())

d = sierpinski(5, 4).fill_color(papaya)
d.render(path, height=256)
st.image(path)
