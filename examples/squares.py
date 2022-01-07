import math
import random

import streamlit as st  # type: ignore

from diagrams import Diagram, RGB, rectangle


def make_square():
    colors = [
        RGB(38, 70, 83),  # charcoal
        RGB(233, 196, 106),  # orange yellow crayola
    ]
    # generate uniformly a value in [-max_angle, max_angle]
    max_angle = math.pi / 24.0
    θ = 2 * max_angle * random.random() - max_angle
    # pick a random color
    i = random.random() > 0.75
    color = colors[i]
    return rectangle(0.15, 0.15).set_stroke_color(color).rotate(θ)


def make_group(num_squares=4):
    return Diagram.concat(make_square() for _ in range(4))


disps = [0.2, 0.4, 0.6, 0.8]
centers = [(x, y) for x in disps for y in disps]
diagram = Diagram.concat(make_group().translate(x, y) for x, y in centers)
diagram = diagram.fmap(lambda s: s.set_stroke_width(0.005))

path = "test.png"
diagram.render(path)
st.image(path)
