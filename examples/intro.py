import streamlit as st  # type: ignore
from colour import Color
from diagrams import *


path = "examples/output/intro-01.png"
papaya = Color("#ff9700")
d = circle(1).fill_color(papaya)
d.render(path)
st.image(path)

path = "examples/output/intro-02.png"
blue = Color("#005FDB")
d = circle(0.5).fill_color(papaya) | square(1).fill_color(blue)
d.render(path)
st.image(path)

path = "examples/output/intro-03.png"
d = hcat(circle(0.1 * i) for i in range(1, 6)).fill_color(blue)
d.render(path)
st.image(path)
