import math

from dataclasses import dataclass
from typing import Any, List, Optional

import cairo
from colour import Color  # type: ignore

from diagrams.bounding_box import BoundingBox
from diagrams.shape import Shape, Circle
from diagrams.style import Style
from diagrams import transform as tx


PyCairoContext = Any
Ident = tx.Identity()


@dataclass
class Diagram:
    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        raise NotImplementedError

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        """Compiles a `Diagram` to a list of `Primitive`s. The transfomation `t`
        is accumulated upwards, from the tree's leaves.

        """
        raise NotImplementedError

    def render(
        self, path: str, height: int = 128, width: Optional[int] = None
    ) -> None:
        pad = 0.05
        box = self.get_bounding_box()

        # infer width to preserve aspect ratio
        width = width or int(height * box.width / box.height)

        # determine scale to fit the largest axis in the target frame size
        if box.width - width <= box.height - height:
            α = height // ((1 + pad) * box.height)
        else:
            α = width // ((1 + pad) * box.width)

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        ctx.scale(α, α)
        ctx.translate(-(1 + pad) * box.tl.x, -(1 + pad) * box.tl.y)

        prims = self.to_list()

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
        box1 = self.get_bounding_box()
        box2 = other.get_bounding_box()
        new_box = box1.union(box2)
        return Compose(new_box, self, other)

    __add__ = atop

    def beside(self, other: "Diagram") -> "Diagram":
        box1 = self.get_bounding_box()
        box2 = other.get_bounding_box()
        dx = box1.right - box2.left
        t = tx.Translate(dx, 0)
        new_box = box1.union(box2.apply_transform(t))
        return Compose(new_box, self, ApplyTransform(t, other))

    __or__ = beside

    def above(self, other: "Diagram") -> "Diagram":
        box1 = self.get_bounding_box()
        box2 = other.get_bounding_box()
        dy = box1.bottom - box2.top
        t = tx.Translate(0, dy)
        new_box = box1.union(box2.apply_transform(t))
        return Compose(new_box, self, ApplyTransform(t, other))

    __truediv__ = above

    def center_xy(self) -> "Diagram":
        box = self.get_bounding_box()
        c = box.center
        t = tx.Translate(-c.x, -c.y)
        return ApplyTransform(t, self)

    def align_t(self) -> "Diagram":
        box = self.get_bounding_box()
        t = tx.Translate(0, -box.top)
        return ApplyTransform(t, self)

    def align_b(self) -> "Diagram":
        box = self.get_bounding_box()
        t = tx.Translate(0, -box.bottom)
        return ApplyTransform(t, self)

    def apply_transform(self, transform: tx.Transform) -> "Diagram":
        return ApplyTransform(transform, self)

    def scale(self, α: float) -> "Diagram":
        return ApplyTransform(tx.Scale(α, α), self)

    def reflect_x(self) -> "Diagram":
        return ApplyTransform(tx.Scale(-1, +1), self)

    def reflect_y(self) -> "Diagram":
        return ApplyTransform(tx.Scale(+1, -1), self)

    # def at(self, x: float, y: float) -> "Diagram":
    #     t = tx.Translate(x, y)
    #     return ApplyTransform(t, self.center_xy())

    def translate(self, dx: float, dy: float) -> "Diagram":
        return ApplyTransform(tx.Translate(dx, dy), self)

    def rotate(self, θ: float) -> "Diagram":
        return ApplyTransform(tx.Rotate(θ), self)

    def rotate_by(self, turns: float) -> "Diagram":
        """Rotate by fractions of a circle (turn)."""
        θ = 2 * math.pi * turns
        return ApplyTransform(tx.Rotate(θ), self)

    def line_width(self, width: float) -> "Diagram":
        return ApplyStyle(Style(line_width=width), self)

    def line_color(self, color: Color) -> "Diagram":
        return ApplyStyle(Style(line_color=color), self)

    def fill_color(self, color: Color) -> "Diagram":
        return ApplyStyle(Style(fill_color=color), self)

    def show_origin(self) -> "Diagram":
        box = self.get_bounding_box()
        origin_size = min(box.height, box.width) / 50
        origin = Primitive(
            Circle(origin_size), Style(fill_color=Color("red")), Ident
        )
        return self + origin


@dataclass
class Primitive(Diagram):
    shape: Shape
    style: Style
    transform: tx.Transform

    @classmethod
    def from_shape(cls, shape: Shape) -> "Primitive":
        return cls(shape, Style.default(), Ident)

    def apply_transform(self, other_transform: tx.Transform) -> "Primitive":
        new_transform = tx.Compose(other_transform, self.transform)
        return Primitive(self.shape, self.style, new_transform)

    def apply_style(self, other_style: Style) -> "Primitive":
        line_width = (
            other_style.line_width
            if other_style.line_width is not None
            else self.style.line_width
        )
        line_color = (
            other_style.line_color
            if other_style.line_color is not None
            else self.style.line_color
        )
        fill_color = (
            other_style.fill_color
            if other_style.fill_color is not None
            else self.style.fill_color
        )
        new_style = Style(
            line_width=line_width, line_color=line_color, fill_color=fill_color
        )
        return Primitive(self.shape, new_style, self.transform)

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        return self.shape.get_bounding_box().apply_transform(t)

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        return [self.apply_transform(t)]


@dataclass
class Empty(Diagram):
    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        return BoundingBox.empty()

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        return []


@dataclass
class Compose(Diagram):
    box: BoundingBox
    diagram1: Diagram
    diagram2: Diagram

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        return self.box.apply_transform(t)

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        return self.diagram1.to_list(t) + self.diagram2.to_list(t)


@dataclass
class ApplyTransform(Diagram):
    transform: tx.Transform
    diagram: Diagram

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        t_new = tx.Compose(t, self.transform)
        return self.diagram.get_bounding_box(t_new)

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        t_new = tx.Compose(t, self.transform)
        return [
            prim.apply_transform(t_new) for prim in self.diagram.to_list(t)
        ]


@dataclass
class ApplyStyle(Diagram):
    style: Style
    diagram: Diagram

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        return self.diagram.get_bounding_box(t)

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        return [
            prim.apply_style(self.style) for prim in self.diagram.to_list(t)
        ]
