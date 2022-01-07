import math

from copy import copy, deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import List, Tuple

from toolz import concat, first, second  # type: ignore

import cairo


Matrix = cairo.Matrix


@dataclass
class Point:
    x: float
    y: float

    def apply(self, transform: "Transform") -> "Point":
        x1, y1 = transform().transform_point(self.x, self.y)
        return Point(x1, y1)


@dataclass
class Extent:
    tl: Point
    br: Point

    def apply(self, transform: "Transform") -> "Extent":
        p = self.tl.apply(transform)
        q = self.br.apply(transform)
        tl = Point(min(p.x, q.x), min(p.y, q.y))
        br = Point(max(p.x, q.x), max(p.y, q.y))
        return Extent(tl, br)

    @staticmethod
    def union(e1, e2):
        tl = Point(min(e1.tl.x, e2.tl.x), min(e1.tl.y, e2.tl.y))
        br = Point(max(e1.br.x, e2.br.x), max(e1.br.y, e2.br.y))
        return Extent(tl, br)

    @staticmethod
    def union_iter(iterable) -> "Extent":
        return reduce(Extent.union, iterable)


class Transform:
    def __call__(self) -> Matrix:
        raise NotImplemented


class Identity(Transform):
    def __call__(self):
        return cairo.Matrix()

    def __str__(self):
        return "Identity"


class Scale(Transform):
    def __init__(self, α):
        self.α = α

    def __call__(self):
        matrix = cairo.Matrix()
        matrix.scale(self.α, self.α)
        return matrix


class ReflectX(Transform):
    def __call__(self):
        matrix = cairo.Matrix()
        matrix.scale(-1, 1)
        return matrix


class ReflectY(Transform):
    def __call__(self):
        matrix = cairo.Matrix()
        matrix.scale(1, -1)
        return matrix


class Rotate(Transform):
    def __init__(self, θ: float):
        self.θ = θ

    def __call__(self):
        return cairo.Matrix.init_rotate(self.θ)


class Translate(Transform):
    def __init__(self, dx: float, dy: float):
        self.dx = dx
        self.dy = dy

    def __call__(self):
        matrix = cairo.Matrix()
        matrix.translate(self.dx, self.dy)
        return matrix


class Compose(Transform):
    def __init__(self, t, u):
        self.t = t
        self.u = u

    def __call__(self):
        # return self.t().multiply(self.u())
        return self.u().multiply(self.t())


class Color:
    def to_float(self):
        raise NotImplemented


@dataclass
class RGB(Color):
    r: int
    g: int
    b: int

    def to_float(self) -> Tuple[float, float, float]:
        return self.r / 255, self.g / 255, self.b / 255


class Primitive:
    def __init__(self):
        self.transform: Transform = Identity()
        # style
        self.fill_color = None
        self.stroke_color = (0.0, 0.0, 0.0)
        self.stroke_width = 0.01

    def rotate(self, θ) -> "Primitive":
        self = deepcopy(self)
        self.transform = Compose(Rotate(θ), self.transform)
        return self

    def scale(self, α) -> "Primitive":
        self = deepcopy(self)
        self.transform = Compose(Scale(α), self.transform)
        return self

    def reflect_x(self) -> "Primitive":
        self = deepcopy(self)
        self.transform = Compose(ReflectX(), self.transform)
        return self

    def reflect_y(self) -> "Primitive":
        self = deepcopy(self)
        self.transform = Compose(ReflectY(), self.transform)
        return self

    def translate(self, dx: float, dy: float) -> "Primitive":
        self = deepcopy(self)
        self.transform = Compose(Translate(dx, dy), self.transform)
        return self

    def set_fill_color(self, color: Color) -> "Primitive":
        self = deepcopy(self)
        self.fill_color = color.to_float()
        return self

    def set_stroke_color(self, color: Color) -> "Primitive":
        self = deepcopy(self)
        self.stroke_color = color.to_float()
        return self

    def set_stroke_width(self, width: float) -> "Primitive":
        self = deepcopy(self)
        self.stroke_width = width
        return self

    def render_shape(self, ctx):
        raise NotImplemented

    def get_extent(self):
        raise NotImplemented

    def render(self, ctx):
        matrix = self.transform()

        ctx.transform(matrix)
        self.render_shape(ctx)

        matrix.invert()
        ctx.transform(matrix)

        # style
        if self.fill_color:
            ctx.set_source_rgb(*self.fill_color)
            ctx.fill_preserve()

        ctx.set_source_rgb(*self.stroke_color)
        ctx.set_line_width(self.stroke_width)
        ctx.stroke()

    def overlay(self, other: "Primitive") -> "Diagram":
        return Diagram([self, other])


class Trail(Primitive):
    """WARN This implementation is different from what is found in `diagrams`"""

    def __init__(self, vertices, extent=None):
        super().__init__()
        self.vertices = vertices
        self.extent = extent
        # TODO concat trails by gluing the end of the first to the start of the second
        # (x0, y0), *_, (xn, yn) = vertices
        # self.start = (x0, y0)
        # self.offset = (xn - x0, yn - y0)

    @classmethod
    def from_offsets(cls, offsets):
        raise NotImplemented

    def render_shape(self, ctx):
        (x0, y0), *rest = self.vertices
        ctx.move_to(x0, y0)
        for x, y in rest:
            ctx.line_to(x, y)

    def get_extent(self):
        if self.extent:
            extent = self.extent
        else:
            tl = Point(min(map(first, self.vertices)), min(map(second, self.vertices)))
            br = Point(max(map(first, self.vertices)), max(map(second, self.vertices)))
            extent = Extent(tl, br)
        return extent.apply(self.transform)


class Blank(Primitive):
    def __init__(self, extent):
        super().__init__()
        self.extent = extent

    def render_shape(self, ctx):
        pass

    def get_extent(self):
        return self.extent.apply(self.transform)


class Circle(Primitive):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius
        self.origin = Point(0, 0)

    def render_shape(self, ctx):
        ctx.arc(self.origin.x, self.origin.y, self.radius, 0, 2 * math.pi)

    def get_extent(self):
        tl = Point(-self.radius, -self.radius)
        br = Point(+self.radius, +self.radius)
        return Extent(tl, br).apply(self.transform)


class Rectangle(Primitive):
    def __init__(self, height: float, width: float):
        super().__init__()
        self.width = width
        self.height = height
        self.origin = Point(0, 0)

    def render_shape(self, ctx):
        left = self.origin.x - self.width / 2
        top = self.origin.y - self.height / 2
        ctx.rectangle(left, top, self.width, self.height)

    def get_extent(self):
        left = self.origin.x - self.width / 2
        top = self.origin.y - self.height / 2
        tl = Point(left, top)
        br = Point(left + self.width, top + self.height)
        return Extent(tl, br).apply(self.transform)


class Diagram:
    def __init__(self, shapes):
        self.shapes = shapes

    @classmethod
    def from_primitive(cls, primitive):
        return cls([primitive])

    @classmethod
    def empty(cls):
        return cls([])

    @classmethod
    def concat(cls, iterable):
        return sum(iterable, cls.empty())

    @staticmethod
    def beside_static(d1, d2):
        e1 = d1.get_extent()
        e2 = d2.get_extent()
        dx1 = e1.br.x
        dx2 = e2.tl.x
        return Diagram(d1.translate(dx2, 0).shapes + d2.translate(dx1, 0).shapes)

    @staticmethod
    def hcat(iterable):
        return reduce(Diagram.beside_static, iterable)

    def get_extent(self):
        return Extent.union_iter(shape.get_extent() for shape in self.shapes)

    def atop(self, other: "Diagram") -> "Diagram":
        return Diagram(self.shapes + other.shapes)

    __add__ = atop

    def beside(self, other: "Diagram") -> "Diagram":
        return Diagram.beside_static(self, other)

    __or__ = beside

    def above(self, other: "Diagram") -> "Diagram":
        e1 = self.get_extent()
        e2 = other.get_extent()
        dy1 = e1.tl.y
        dy2 = e2.br.y
        return Diagram(self.translate(0, dy2).shapes + other.translate(0, dy1).shapes)

    __truediv__ = above

    def show_origin(self):
        o = Circle(0.005).set_stroke_color(RGB(255, 0, 0))
        return self + Diagram.from_primitive(o)

    def translate(self, dx, dy):
        return self.fmap(lambda shape: shape.translate(dx, dy))

    def reflect_x(self):
        return self.fmap(lambda shape: shape.reflect_x())

    def reflect_y(self):
        return self.fmap(lambda shape: shape.reflect_y())

    def rotate(self, θ):
        return self.fmap(lambda shape: shape.rotate(θ))

    def scale(self, α):
        return self.fmap(lambda shape: shape.scale(α))

    def set_stroke_width(self, w):
        return self.fmap(lambda shape: shape.set_stroke_width(w))

    def set_stroke_color(self, color):
        return self.fmap(lambda shape: shape.set_stroke_color(color))

    def set_fill_color(self, color):
        return self.fmap(lambda shape: shape.set_fill_color(color))

    def fmap(self, f) -> "Diagram":
        return Diagram([f(shape) for shape in self.shapes])

    def render(self, path):
        WIDTH, HEIGHT = 512, 512

        pad = 1.05
        extent = self.get_extent()

        width = extent.br.x - extent.tl.x
        height = extent.br.y - extent.tl.y
        size = max(width, height)

        α = WIDTH / (pad * size)

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
        ctx = cairo.Context(surface)

        ctx.scale(α, α)
        ctx.translate(-pad * extent.tl.x, -pad * extent.tl.y)

        for shape in self.shapes:
            shape.render(ctx)

        surface.write_to_png(path)


circle = lambda size: Diagram.from_primitive(Circle(size))
rectangle = lambda height, width: Diagram.from_primitive(Rectangle(height, width))
square = lambda size: rectangle(size, size)
