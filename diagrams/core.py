from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cairo
from colour import Color


from diagrams.bounding_box import BoundingBox
from diagrams.point import Point, ORIGIN
from diagrams.shape import Shape
from diagrams import transform as tx


PyCairoContext = Any
I = tx.Identity()


@dataclass
class Style:
    def __init__(self, line_color: Optional[Color] = None, fill_color: Optional[Color] = None):
        self.line_color = line_color
        self.fill_color = fill_color

    def render(self, ctx: PyCairoContext) -> None:
        if self.fill_color:
            ctx.set_source_rgb(*self.fill_color.rgb)
            ctx.fill_preserve()

        ctx.set_source_rgb(*self.line_color.rgb)
        ctx.set_line_width(0.01)
        ctx.stroke()


@dataclass
class Diagram:
    def get_bounding_box(self, t: tx.Transform = I) -> BoundingBox:
        raise NotImplemented

    def to_list(self, t: tx.Transform = I) -> List["Primitive"]:
        raise NotImplemented

    def render(self, path: str, width: int=128, height: int=128, pad: float=0.05) -> None:
        box = self.get_bounding_box()

        size = max(box.width, box.height)
        α = width // ((1 + pad) * size)

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
        dx = box1.right + box2.right
        t = tx.Translate(dx, 0)
        new_box = box1.union(box2.apply_transform(t))
        return Compose(new_box, self, ApplyTransform(t, other))

    __or__ = beside

    def apply_transform(self, transform: tx.Transform) -> "Diagram":
        return ApplyTransform(transform, self)

    def rotate(self, θ: float) -> "Diagram":
        return ApplyTransform(tx.Rotate(θ), self)

    def fill_color(self, color: Color) -> "Diagram":
        return ApplyStyle(Style(fill_color=color), self)


@dataclass
class Primitive(Diagram):
    shape: Shape
    style: Style
    transform: tx.Transform

    @classmethod
    def from_shape(cls, shape: Shape) -> "Primitive":
        return cls(shape, Style(line_color=Color("black")), I)

    def apply_transform(self, other_transform: tx.Transform) -> "Primitive":
        new_transform = tx.Compose(self.transform, other_transform)
        return Primitive(self.shape, self.style, new_transform)

    def apply_style(self, other_style: Style) -> "Primitive":
        line_color = other_style.line_color or self.style.line_color
        fill_color = other_style.fill_color or self.style.fill_color
        new_style = Style(line_color=line_color, fill_color=fill_color)
        return Primitive(self.shape, new_style, self.transform)

    def get_bounding_box(self, t: tx.Transform = I) -> BoundingBox:
        return self.shape.get_bounding_box().apply_transform(t)

    def to_list(self, t: tx.Transform = I) -> List["Primitive"]:
        return [self.apply_transform(t)]


@dataclass
class Empty(Diagram):
    def get_bounding_box(self, t: tx.Transform = I) -> BoundingBox:
        return BoundingBox.empty()

    def to_list(self, t: tx.Transform = I) -> List["Primitive"]:
        return []


@dataclass
class Compose(Diagram):
    box: BoundingBox
    diagram1: Diagram
    diagram2: Diagram

    def get_bounding_box(self, t: tx.Transform = I) -> BoundingBox:
        return self.box
        # box1 = self.diagram1.get_bounding_box(t)
        # box2 = self.diagram2.get_bounding_box(t)
        # return box1.union(box2)

    def to_list(self, t: tx.Transform = I) -> List["Primitive"]:
        return self.diagram1.to_list(t) + self.diagram2.to_list(t)


@dataclass
class ApplyTransform(Diagram):
    transform: tx.Transform
    diagram: Diagram

    def get_bounding_box(self, t: tx.Transform = I) -> BoundingBox:
        t_new = tx.Compose(self.transform, t)
        return self.diagram.get_bounding_box(t_new)

    def to_list(self, t: tx.Transform = I) -> List["Primitive"]:
        return [prim.apply_transform(tx.Compose(self.transform, t)) for prim in self.diagram.to_list(t)]


@dataclass
class ApplyStyle(Diagram):
    style: Style
    diagram: Diagram

    def get_bounding_box(self, t: tx.Transform = I) -> BoundingBox:
        return self.diagram.get_bounding_box(t)

    def to_list(self, t: tx.Transform = I) -> List["Primitive"]:
        return [prim.apply_style(self.style) for prim in self.diagram.to_list(t)]
