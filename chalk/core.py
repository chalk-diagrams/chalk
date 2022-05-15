from dataclasses import dataclass
from typing import Any, List, Optional

import cairo
import svgwrite
from svgwrite import Drawing
from svgwrite.base import BaseElement
from colour import Color

from chalk.bounding_box import BoundingBox
from chalk.shape import Shape, Circle, Rectangle
from chalk.style import Style
from chalk import transform as tx


PyCairoContext = Any
Ident = tx.Identity()


@dataclass
class Diagram(tx.Transformable):
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

    def render_svg(
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
        dwg = svgwrite.Drawing(
            path,
            size=(width, height),
        )
        x, y = -(1 + pad) * box.tl.x, -(1 + pad) * box.tl.y
        outer = dwg.g(
            transform=f"scale({α}) translate({x} {y})",
            style="fill:white; stroke: black; stroke-width: 0.01;",
        )
        # Arrow marker
        marker = dwg.marker(
            id="arrow", refX=5.0, refY=1.7, size=(5, 3.5), orient="auto"
        )
        marker.add(dwg.polygon([(0, 0), (5, 1.75), (0, 3.5)]))
        dwg.defs.add(marker)

        dwg.add(outer)
        outer.add(self.to_svg(dwg, Style.default()))
        dwg.save()

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

    def above2(self, other: "Diagram") -> "Diagram":
        box1 = self.get_bounding_box()
        box2 = other.get_bounding_box()
        dy = box1.bottom - box2.top
        t = tx.Translate(0, -dy)
        new_box = box2.union(box1.apply_transform(t))
        return Compose(new_box, ApplyTransform(t, self), other)

    __floordiv__ = above2

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

    def align_r(self) -> "Diagram":
        box = self.get_bounding_box()
        t = tx.Translate(-box.right, 0)
        return ApplyTransform(t, self)

    def align_l(self) -> "Diagram":
        box = self.get_bounding_box()
        t = tx.Translate(-box.left, 0)
        return ApplyTransform(t, self)

    def align_tl(self) -> "Diagram":
        return self.align_t().align_l()

    def align_br(self) -> "Diagram":
        return self.align_b().align_r()

    def align_tr(self) -> "Diagram":
        return self.align_t().align_r()

    def align_bl(self) -> "Diagram":
        return self.align_b().align_l()

    def pad_l(self, extra: float) -> "Diagram":
        box = self.get_bounding_box()
        new_box = BoundingBox.from_limits(
            box.tl.x - extra, box.tl.y, box.br.x, box.br.y
        )
        return Compose(new_box, self, Empty())

    def pad_t(self, extra: float) -> "Diagram":
        box = self.get_bounding_box()
        new_box = BoundingBox.from_limits(
            box.tl.x, box.tl.y - extra, box.br.x, box.br.y
        )
        return Compose(new_box, self, Empty())

    def pad_r(self, extra: float) -> "Diagram":
        box = self.get_bounding_box()
        new_box = BoundingBox.from_limits(
            box.tl.x, box.tl.y, box.br.x + extra, box.br.y
        )
        return Compose(new_box, self, Empty())

    def pad_b(self, extra: float) -> "Diagram":
        box = self.get_bounding_box()
        new_box = BoundingBox.from_limits(
            box.tl.x, box.tl.y, box.br.x, box.br.y + extra
        )
        return Compose(new_box, self, Empty())

    def scale_uniform_to_x(self, x: float) -> "Diagram":
        box = self.get_bounding_box()
        α = x / box.width
        return ApplyTransform(tx.Scale(α, α), self)

    def scale_uniform_to_y(self, y: float) -> "Diagram":
        box = self.get_bounding_box()
        α = y / box.height
        return ApplyTransform(tx.Scale(α, α), self)

    def apply_transform(self, t: tx.Transform) -> "Diagram":  # type: ignore
        return ApplyTransform(t, self)

    # def at(self, x: float, y: float) -> "Diagram":
    #     t = tx.Translate(x, y)
    #     return ApplyTransform(t, self.center_xy())

    def line_width(self, width: float) -> "Diagram":
        return ApplyStyle(Style(line_width=width), self)

    def line_color(self, color: Color) -> "Diagram":
        return ApplyStyle(Style(line_color=color), self)

    def fill_color(self, color: Color) -> "Diagram":
        return ApplyStyle(Style(fill_color=color), self)

    def fill_opacity(self, opacity: float) -> "Diagram":
        return ApplyStyle(Style(fill_opacity=opacity), self)

    def dashing(
        self, dashing_strokes: List[float], offset: float
    ) -> "Diagram":
        return ApplyStyle(Style(dashing=(dashing_strokes, offset)), self)

    def at_center(self, other: "Diagram") -> "Diagram":
        box1 = self.get_bounding_box()
        box2 = other.get_bounding_box()
        c = box1.center
        t = tx.Translate(c.x, c.y)
        new_box = box1.union(box2.apply_transform(t))
        return Compose(new_box, self, ApplyTransform(t, other))

    def show_origin(self) -> "Diagram":
        box = self.get_bounding_box()
        origin_size = min(box.height, box.width) / 50
        origin = Primitive(
            Circle(origin_size), Style(fill_color=Color("red")), Ident
        )
        return self + origin

    def show_bounding_box(self) -> "Diagram":
        box = self.get_bounding_box()
        origin = Primitive(
            Rectangle(box.width, box.height),
            Style(fill_opacity=0, line_color=Color("red")),
            Ident,
        ).translate(box.center.x, box.center.y)
        return self + origin

    def named(self, name: str) -> "Diagram":
        return ApplyName(name, self)

    def get_subdiagram_bounding_box(
        self, name: str, t: tx.Transform = Ident
    ) -> Optional[BoundingBox]:
        return None

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        raise NotImplementedError


@dataclass
class Primitive(Diagram):
    shape: Shape
    style: Style
    transform: tx.Transform

    @classmethod
    def from_shape(cls, shape: Shape) -> "Primitive":
        return cls(shape, Style.default(), Ident)

    def apply_transform(self, t: tx.Transform) -> "Primitive":  # type: ignore
        new_transform = tx.Compose(t, self.transform)
        return Primitive(self.shape, self.style, new_transform)

    def apply_style(self, other_style: Style) -> "Primitive":
        return Primitive(
            self.shape, self.style.merge(other_style), self.transform
        )

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        new_transform = tx.Compose(t, self.transform)
        return self.shape.get_bounding_box().apply_transform(new_transform)

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        return [self.apply_transform(t)]

    def to_svg(self, dwg: Drawing, other_style: Style) -> BaseElement:
        style = self.style.merge(other_style).to_svg()
        transform = self.transform.to_svg()
        inner = self.shape.render_svg(dwg)

        if not style and not transform:
            return inner
        else:
            if not style:
                style = ";"
            g = dwg.g(transform=transform, style=style)
            g.add(inner)
            return g


@dataclass
class Empty(Diagram):
    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        return BoundingBox.empty()

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        return []

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        return dwg.g()


@dataclass
class Compose(Diagram):
    box: BoundingBox
    diagram1: Diagram
    diagram2: Diagram

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        return self.box.apply_transform(t)

    def get_subdiagram_bounding_box(
        self, name: str, t: tx.Transform = Ident
    ) -> Optional[BoundingBox]:
        bb = self.diagram1.get_subdiagram_bounding_box(name, t)
        if bb is None:
            bb = self.diagram2.get_subdiagram_bounding_box(name, t)
        return bb

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        return self.diagram1.to_list(t) + self.diagram2.to_list(t)

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        g = dwg.g()
        g.add(self.diagram1.to_svg(dwg, style))
        g.add(self.diagram2.to_svg(dwg, style))
        return g


@dataclass
class ApplyTransform(Diagram):
    transform: tx.Transform
    diagram: Diagram

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        t_new = tx.Compose(t, self.transform)
        return self.diagram.get_bounding_box(t_new)

    def get_subdiagram_bounding_box(
        self, name: str, t: tx.Transform = Ident
    ) -> Optional[BoundingBox]:
        t_new = tx.Compose(t, self.transform)
        return self.diagram.get_subdiagram_bounding_box(name, t_new)

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        t_new = tx.Compose(t, self.transform)
        return [
            prim.apply_transform(t_new) for prim in self.diagram.to_list(t)
        ]

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        g = dwg.g(transform=self.transform.to_svg())
        g.add(self.diagram.to_svg(dwg, style))
        return g


@dataclass
class ApplyStyle(Diagram):
    style: Style
    diagram: Diagram

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        return self.diagram.get_bounding_box(t)

    def get_subdiagram_bounding_box(
        self, name: str, t: tx.Transform = Ident
    ) -> Optional[BoundingBox]:
        return self.diagram.get_subdiagram_bounding_box(name, t)

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        return [
            prim.apply_style(self.style) for prim in self.diagram.to_list(t)
        ]

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        return self.diagram.to_svg(dwg, self.style.merge(style))


@dataclass
class ApplyName(Diagram):
    dname: str
    diagram: Diagram

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        return self.diagram.get_bounding_box(t)

    def get_subdiagram_bounding_box(
        self, name: str, t: tx.Transform = Ident
    ) -> Optional[BoundingBox]:
        if name == self.dname:
            return self.diagram.get_bounding_box(t)
        else:
            return None

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        return [prim for prim in self.diagram.to_list(t)]

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        g = dwg.g()
        g.add(self.diagram.to_svg(dwg, style))
        return g
