import pdb
import pprint

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Any, List, Optional, Tuple

import cairo

from diagrams.bounding_box import BoundingBox, EMPTY_BOUNDING_BOX
from diagrams.shape import Shape, Circle, Rectangle
from diagrams.point import Point, ORIGIN
from diagrams import transform as tx


PyCairoContext = Any


@dataclass
class Style:
    line_color: Tuple[float, float, float]

    def render(self, ctx: PyCairoContext) -> None:
        ctx.set_source_rgb(*self.line_color)
        ctx.set_line_width(0.01)
        ctx.stroke()


@dataclass
class Diagram(ABCMeta):
    @abstractmethod
    def get_bounding_box(self) -> BoundingBox:
        pass

    @abstractmethod
    def to_list(self) -> List["Primitive"]:
        pass

    def render(self, path: str, width: int=128, height: int=128, pad: float=0.05) -> None:
        box = self.get_bounding_box()

        size = max(box.width, box.height)
        α = width // ((1 + pad) * size)

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        ctx.scale(α, α)
        ctx.translate(-(1 + pad) * box.tl.x, -(1 + pad) * box.tl.y)

        prims = self.to_list()
        pprint.pprint(prims)

        for prim in prims:
            # apply transformation
            matrix = prim.transform()
            ctx.transform(matrix)

            prim.shape.render(ctx)
            prim.style.render(ctx)

            # undo transformation
            matrix.invert()
            ctx.transform(matrix)

        surface.write_to_png(path)

    def atop(self, other: "Diagram") -> "Diagram":
        box = self.get_bounding_box().union(other.get_bounding_box())
        return Compose(box, self, other)

    __add__ = atop

    def beside(self, other: "Diagram") -> "Diagram":
        # TODO? Transform bounding boxes?!
        box1 = self.get_bounding_box()
        box2 = other.get_bounding_box()
        dx = box1.right + box2.right
        top = min(box1.top, box2.top)
        bot = max(box1.bottom, box2.bottom)
        new_box = BoundingBox.from_limits(box1.left, top, box2.right + dx, bot)
        return Compose(new_box, self, ApplyTransform(tx.Translate(dx, 0), other))

    __or__ = beside

    def apply_transform(self, transform: tx.Transform) -> "Diagram":
        return ApplyTransform(transform, self)


@dataclass
class Primitive(Diagram):
    shape: Shape
    style: Style
    transform: tx.Transform

    @classmethod
    def from_shape(cls, shape: Shape) -> "Primitive":
        return cls(shape, Style(line_color=(0, 0, 0)), tx.Identity())

    def apply_transform(self, other_transform: tx.Transform) -> "Primitive":
        return Primitive(
            self.shape, self.style, tx.Compose(self.transform, other_transform)
        )

    def get_bounding_box(self) -> BoundingBox:
        return self.shape.get_bounding_box()

    def to_list(self) -> List["Primitive"]:
        return [self]


@dataclass
class Empty(Diagram):
    def get_bounding_box(self) -> BoundingBox:
        return EMPTY_BOUNDING_BOX

    def to_list(self) -> List["Primitive"]:
        return []


@dataclass
class Compose(Diagram):
    box: BoundingBox
    diagram1: Diagram
    diagram2: Diagram

    def get_bounding_box(self) -> BoundingBox:
        return self.box

    def to_list(self) -> List["Primitive"]:
        return self.diagram1.to_list() + self.diagram2.to_list()


@dataclass
class ApplyTransform(Diagram):
    transform: tx.Transform
    diagram: Diagram

    def get_bounding_box(self) -> BoundingBox:
        return self.diagram.get_bounding_box().apply_transform(self.transform)

    def to_list(self) -> List["Primitive"]:
        return [prim.apply_transform(self.transform) for prim in self.diagram.to_list()]


@dataclass
class ApplyStyle(Diagram):
    style: Style
    diagram: Diagram

    def get_bounding_box(self) -> BoundingBox:
        return self.diagram.get_bounding_box()

    def to_list(self) -> List["Primitive"]:
        return self.diagram.to_list()


def circle(size: float) -> Diagram:
    return Primitive.from_shape(Circle(size))


def square(size: float) -> Diagram:
    return Primitive.from_shape(Rectangle(size, size))


def beside(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    return diagram1.beside(diagram2)


def hcat(diagrams: List[Diagram]) -> Diagram:
    return reduce(beside, diagrams)


if __name__ == "__main__":
    import streamlit as st  # type: ignore

    path = "/tmp/o.png"
    # example 1
    # d = circle(1) + square(1)
    # d.render(path)
    # st.code("circle(1) + square(1)")
    # st.image(path)
    # example 2
    # d = circle(1) | square(1)
    # st.code("circle(1) | square(1)")
    # st.code(repr(d))
    # d.render(path)
    # st.image(path)
    # example 3
    d = hcat([circle(1) for _ in range(6)])
    st.code(repr(d))
    d.render(path)
    st.image(path)
