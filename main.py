# Design goals:
# - Add an abstraction layer over PyCairo
# - Replace the imperative API with a more declarative design (maybe similar to the `diagrams` library from Haskell)
#
# TODO
# - [ ] Allow change of backend for the `render` function
# - [ ] Update width and height of image to fit all the contents
#

import math

from typing import List, Tuple

import cairo
import streamlit as st


PATH = "test.png"


class Primitive:
    def __init__(self):
        self.fill_color = None
        self.stroke_color = (0, 0, 0)
        self.stroke_width = 0.01

    def set_fill_color(self, r: float, g: float, b: float) -> "Primitive":
        self.fill_color = r, g, b
        return self

    def set_stroke_color(self, r: float, g: float, b: float) -> "Primitive":
        self.stroke_color = r, g, b
        return self

    def render_shape(self, ctx):
        raise NotImplemented

    def render(self, ctx):
        self.render_shape(ctx)

        # style
        if self.fill_color:
            ctx.set_source_rgb(*self.fill_color)
            ctx.fill_preserve()

        ctx.set_source_rgb(*self.stroke_color)
        ctx.set_line_width(self.stroke_width)
        ctx.stroke()

    def overlay(self, other: "Primitive") -> "Diagram":
        return Diagram([self, other])


class Circle(Primitive):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius
        self.center = (0.0, 0.0)  # type: Tuple[float, float]

    def at(self, x: float, y: float) -> "Circle":
        self.center = x, y
        return self

    def render_shape(self, ctx):
        ctx.arc(*self.center, self.radius, 0, 2 * math.pi)


class Rectangle(Primitive):
    def __init__(self, height: float, width: float):
        super().__init__()
        self.height = height
        self.width = width
        self.left = 0.0
        self.top = 0.0

    def at(self, left: float, top: float) -> "Rectangle":
        self.left = left
        self.top = top
        return self

    def render_shape(self, ctx):
        ctx.rectangle(self.left, self.top, self.width, self.height)


class Diagram:
    def __init__(self, shapes: List[Primitive]):
        self.shapes = shapes

    def add(self, shape: Primitive) -> "Diagram":
        self.shapes = self.shapes + [shape]
        return self

    def fmap(self, f) -> "Diagram":
        return Diagram([f(shape) for shape in self.shapes])

    def render(self):
        WIDTH, HEIGHT = 256, 256
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
        ctx = cairo.Context(surface)
        ctx.scale(WIDTH, HEIGHT)

        for shape in self.shapes:
            shape.render(ctx)

        surface.write_to_png(PATH)


circle1 = Circle(0.1).at(0.2, 0.2)
circle2 = Circle(0.2).at(0.5, 0.5)
rect = (
    Rectangle(0.3, 0.3)
    .at(0.2, 0.2)
    .set_fill_color(0.3, 0.3, 0.3)
    .set_stroke_color(1, 0, 0)
)

diagram = circle1.overlay(circle2)
diagram = diagram.fmap(lambda s: s.set_stroke_color(0, 0, 1))
diagram = diagram.add(rect)
diagram.render()

st.image(PATH)
