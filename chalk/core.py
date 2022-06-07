from dataclasses import dataclass
from typing import Any, List, Optional

import cairo
import svgwrite
import tempfile
import os
from svgwrite import Drawing
from svgwrite.base import BaseElement
from colour import Color

from chalk.bounding_box import BoundingBox
from chalk.shape import Shape, Circle, Rectangle
from chalk.style import Style
from chalk.utils import imgen
from chalk import transform as tx


PyCairoContext = Any
Ident = tx.Identity()


@dataclass
class Diagram(tx.Transformable):
    """Diagram class."""

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        """Get the bounding box of a diagram."""
        raise NotImplementedError

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        """Compiles a `Diagram` to a list of `Primitive`s. The transfomation `t`
        is accumulated upwards, from the tree's leaves.
        """
        raise NotImplementedError

    def display(
        self, height: int = 256, verbose: bool = True, **kwargs: Any
    ) -> None:
        """Display the diagram using the default renderer.

        Note: see ``chalk.utils.imgen`` for details on the keyword arguments.
        """
        # update kwargs with defaults and user-specified values
        kwargs.update({"height": height})
        kwargs.update({"verbose": verbose})
        kwargs.update({"dirpath": None})
        kwargs.update({"wait": kwargs.get("wait", 1)})
        # render and display the diagram
        imgen(self, **kwargs)

    def render(
        self, path: str, height: int = 128, width: Optional[int] = None
    ) -> None:
        """Render the diagram to a PNG file.

        Args:
            path (str): Path of the .png file.
            height (int, optional): Height of the rendered image.
                                    Defaults to 128.
            width (Optional[int], optional): Width of the rendered image.
                                             Defaults to None.
        """
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
        """Render the diagram to an SVG file.

        Args:
            path (str): Path of the .svg file.
            height (int, optional): Height of the rendered image.
                                    Defaults to 128.
            width (Optional[int], optional): Width of the rendered image.
                                             Defaults to None.
        """
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

    def _repr_svg_(self) -> str:
        f = tempfile.NamedTemporaryFile(delete=False)
        self.render_svg(f.name)
        f.close()
        svg = open(f.name).read()
        os.unlink(f.name)
        return svg

    def atop(self, other: "Diagram") -> "Diagram":
        """Given two diagrams ``a`` and ``b``, ``a.atop(b)``
        places ``a`` and ``b`` such that none of them move.
        This is equivalent to **union** operation of two the
        shapes.

        💡 ``a.atop(b)`` is equivalent to ``a + b``.

        Args:
            other (Diagram): Another diagram object.

        Returns:
            Diagram: A diagram object.
        """
        box1 = self.get_bounding_box()
        box2 = other.get_bounding_box()
        new_box = box1.union(box2)
        return Compose(new_box, self, other)

    __add__ = atop

    def beside(self, other: "Diagram") -> "Diagram":
        """Given two diagrams ``a`` and ``b``, ``a.beside(b)``
        places ``b`` on the right side of ``a``. This moves
        ``b`` up to touch ``a``.

        💡 ``a.beside(b)`` is equivalent to ``a | b``.

        Args:
            other (Diagram): Another diagram object.

        Returns:
            Diagram: A diagram object.
        """
        box1 = self.get_bounding_box()
        box2 = other.get_bounding_box()
        dx = box1.right - box2.left
        t = tx.Translate(dx, 0)
        new_box = box1.union(box2.apply_transform(t))
        return Compose(new_box, self, ApplyTransform(t, other))

    __or__ = beside

    def above(self, other: "Diagram") -> "Diagram":
        """Given two diagrams ``a`` and ``b``, ``a.above(b)``
        places ``b`` under ``a``. This moves ``b`` up to
        touch ``a``.

        💡 ``a.above(b)`` is equivalent to ``a / b``.

        Args:
            other (Diagram): Another diagram object.

        Returns:
            Diagram: A diagram object.
        """
        box1 = self.get_bounding_box()
        box2 = other.get_bounding_box()
        dy = box1.bottom - box2.top
        t = tx.Translate(0, dy)
        new_box = box1.union(box2.apply_transform(t))
        return Compose(new_box, self, ApplyTransform(t, other))

    __truediv__ = above

    def above2(self, other: "Diagram") -> "Diagram":
        """Given two diagrams ``a`` and ``b``, ``a.above2(b)``
        places ``a`` on top of ``b``. This moves ``a`` down to
        touch ``b``.

        💡 ``a.above2(b)`` is equivalent to ``a // b``.

        Args:
            other (Diagram): Another diagram object.

        Returns:
            Diagram: A diagram object.
        """
        box1 = self.get_bounding_box()
        box2 = other.get_bounding_box()
        dy = box1.bottom - box2.top
        t = tx.Translate(0, -dy)
        new_box = box2.union(box1.apply_transform(t))
        return Compose(new_box, ApplyTransform(t, self), other)

    __floordiv__ = above2

    def center_xy(self) -> "Diagram":
        """Center a diagram.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        c = box.center
        t = tx.Translate(-c.x, -c.y)
        return ApplyTransform(t, self)

    def align_t(self) -> "Diagram":
        """Align a diagram with its top edge.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        t = tx.Translate(0, -box.top)
        return ApplyTransform(t, self)

    def align_b(self) -> "Diagram":
        """Align a diagram with its bottom edge.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        t = tx.Translate(0, -box.bottom)
        return ApplyTransform(t, self)

    def align_r(self) -> "Diagram":
        """Align a diagram with its right edge.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        t = tx.Translate(-box.right, 0)
        return ApplyTransform(t, self)

    def align_l(self) -> "Diagram":
        """Align a diagram with its left edge.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        t = tx.Translate(-box.left, 0)
        return ApplyTransform(t, self)

    def align_tl(self) -> "Diagram":
        """Align a diagram with its top-left edges.

        Returns:
            Diagram: A diagram object.
        """
        return self.align_t().align_l()

    def align_br(self) -> "Diagram":
        """Align a diagram with its bottom-right edges.

        Returns:
            Diagram: A diagram object.
        """
        return self.align_b().align_r()

    def align_tr(self) -> "Diagram":
        """Align a diagram with its top-right edges.

        Returns:
            Diagram: A diagram object.
        """
        return self.align_t().align_r()

    def align_bl(self) -> "Diagram":
        """Align a diagram with its bottom-left edges.

        Returns:
            Diagram: A diagram object.
        """
        return self.align_b().align_l()

    def pad_l(self, extra: float) -> "Diagram":
        """Add outward directed left-side padding for
        a diagram. This padding is applied **only** on
        the **left** side.

        Args:
            extra (float): Amount of padding to add.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        new_box = BoundingBox.from_limits(
            box.tl.x - extra, box.tl.y, box.br.x, box.br.y
        )
        return Compose(new_box, self, Empty())

    def pad_t(self, extra: float) -> "Diagram":
        """Add outward directed top-side padding for
        a diagram. This padding is applied **only** on
        the **top** side.

        Args:
            extra (float): Amount of padding to add.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        new_box = BoundingBox.from_limits(
            box.tl.x, box.tl.y - extra, box.br.x, box.br.y
        )
        return Compose(new_box, self, Empty())

    def pad_r(self, extra: float) -> "Diagram":
        """Add outward directed right-side padding for
        a diagram. This padding is applied **only** on
        the **right** side.

        Args:
            extra (float): Amount of padding to add.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        new_box = BoundingBox.from_limits(
            box.tl.x, box.tl.y, box.br.x + extra, box.br.y
        )
        return Compose(new_box, self, Empty())

    def pad_b(self, extra: float) -> "Diagram":
        """Add outward directed bottom-side padding for
        a diagram. This padding is applied **only** on
        the **bottom** side.

        Args:
            extra (float): Amount of padding to add.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        new_box = BoundingBox.from_limits(
            box.tl.x, box.tl.y, box.br.x, box.br.y + extra
        )
        return Compose(new_box, self, Empty())

    def pad(self, extra: float) -> "Diagram":
        """Add outward directed padding for a diagram.
        This padding is applied uniformly on all sides.

        Args:
            extra (float): Amount of padding to add.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        new_box = BoundingBox.from_limits(
            box.tl.x - extra,
            box.tl.y - extra,
            box.br.x + extra,
            box.br.y + extra,
        )
        return Compose(new_box, self, Empty())

    def scale_uniform_to_x(self, x: float) -> "Diagram":
        """Apply uniform scaling along the x-axis.

        Args:
            x (float): Amount of scaling along the x-axis.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        α = x / box.width
        return ApplyTransform(tx.Scale(α, α), self)

    def scale_uniform_to_y(self, y: float) -> "Diagram":
        """Apply uniform scaling along the y-axis.

        Args:
            y (float): Amount of scaling along the y-axis.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        α = y / box.height
        return ApplyTransform(tx.Scale(α, α), self)

    def apply_transform(self, t: tx.Transform) -> "Diagram":  # type: ignore
        """Apply a transformation.

        Args:
            t (tx.Transform): A transformation.

        Returns:
            Diagram: A diagram object.
        """
        return ApplyTransform(t, self)

    # def at(self, x: float, y: float) -> "Diagram":
    #     t = tx.Translate(x, y)
    #     return ApplyTransform(t, self.center_xy())

    def line_width(self, width: float) -> "Diagram":
        """Apply specified line-width to the edge of
        the diagram.

        Args:
            width (float): Amount of width.

        Returns:
            Diagram: A diagram object.
        """
        return ApplyStyle(Style(line_width=width), self)

    def line_color(self, color: Color) -> "Diagram":
        """Apply specified line-color to the edge of
        the diagram.

        Args:
            color (float): A color (``colour.Color``).

        Returns:
            Diagram: A diagram object.
        """
        return ApplyStyle(Style(line_color=color), self)

    def fill_color(self, color: Color) -> "Diagram":
        """Apply specified fill-color to the diagram.

        Args:
            color (Color): A color object.

        Returns:
            Diagram: A diagram object.
        """
        return ApplyStyle(Style(fill_color=color), self)

    def fill_opacity(self, opacity: float) -> "Diagram":
        """Apply specified amount of opacity to the diagram.

        Args:
            opacity (float): Amount of opacity (between 0 and 1).

        Returns:
            Diagram: A diagram object.
        """
        return ApplyStyle(Style(fill_opacity=opacity), self)

    def dashing(
        self, dashing_strokes: List[float], offset: float
    ) -> "Diagram":
        """Apply dashed line to the edge of a diagram.

        > [TODO]: improve args description.

        Args:
            dashing_strokes (List[float]): Dashing strokes
            offset (float): Amount of offset

        Returns:
            Diagram: A diagram object.
        """
        return ApplyStyle(Style(dashing=(dashing_strokes, offset)), self)

    def at_center(self, other: "Diagram") -> "Diagram":
        """Center two given diagrams.

        💡 `a.at_center(b)` means center of ``a`` is translated
        to the center of ``b``, and ``b`` sits on top of
        ``a`` along the axis out of the plane of the image.

        💡 In other words, ``b`` occludes ``a``.

        Args:
            other (Diagram): Another diagram object.

        Returns:
            Diagram: A diagram object.
        """
        box1 = self.get_bounding_box()
        box2 = other.get_bounding_box()
        c = box1.center
        t = tx.Translate(c.x, c.y)
        new_box = box1.union(box2.apply_transform(t))
        return Compose(new_box, self, ApplyTransform(t, other))

    def show_origin(self) -> "Diagram":
        """Show a ``red`` dot at the origin of a diagram.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        origin_size = min(box.height, box.width) / 50
        origin = Primitive(
            Circle(origin_size), Style(fill_color=Color("red")), Ident
        )
        return self + origin

    def show_bounding_box(self) -> "Diagram":
        """Show ``red`` bounding box of a diagram.

        Returns:
            Diagram: A diagram object.
        """
        box = self.get_bounding_box()
        origin = Primitive(
            Rectangle(box.width, box.height),
            Style(fill_opacity=0, line_color=Color("red")),
            Ident,
        ).translate(box.center.x, box.center.y)
        return self + origin

    def named(self, name: str) -> "Diagram":
        """Add a name to a diagram.

        Args:
            name (str): Diagram name.

        Returns:
            Diagram: A diagram object.
        """
        return ApplyName(name, self)

    def get_subdiagram_bounding_box(
        self, name: str, t: tx.Transform = Ident
    ) -> Optional[BoundingBox]:
        """Get the bounding box of the sub-diagram."""
        return None

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Convert a diagram to SVG image."""
        raise NotImplementedError


@dataclass
class Primitive(Diagram):
    """Primitive class.

    This is derived from a ``chalk.core.Diagram`` class.

    [TODO]: explain what Primitive class is for.
    """

    shape: Shape
    style: Style
    transform: tx.Transform

    @classmethod
    def from_shape(cls, shape: Shape) -> "Primitive":
        """Create and return a primitive from a shape.

        Args:
            shape (Shape): A shape object.

        Returns:
            Primitive: A primitive object.
        """
        return cls(shape, Style.default(), Ident)

    def apply_transform(self, t: tx.Transform) -> "Primitive":  # type: ignore
        """Applies a transform and returns a primitive.

        Args:
            t (Transform): A transform object.

        Returns:
            Primitive: A primitive object.
        """
        new_transform = tx.Compose(t, self.transform)
        return Primitive(self.shape, self.style, new_transform)

    def apply_style(self, other_style: Style) -> "Primitive":
        """Applies a style and returns a primitive.

        Args:
            other_style (Style): A style object.

        Returns:
            Primitive: A primitive object.
        """
        return Primitive(
            self.shape, self.style.merge(other_style), self.transform
        )

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        """Apply a transform and return a bounding box.

        Args:
            t (Transform): A transform object
                           Defaults to Ident.

        Returns:
            BoundingBox: A bounding box object.
        """
        new_transform = tx.Compose(t, self.transform)
        return self.shape.get_bounding_box().apply_transform(new_transform)

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        """Returns a list of primitives.

        Args:
            t (Transform): A transform object
                           Defaults to Ident.

        Returns:
            List[Primitive]: List of primitives.
        """
        return [self.apply_transform(t)]

    def to_svg(self, dwg: Drawing, other_style: Style) -> BaseElement:
        """Convert a diagram to SVG image."""
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
    """An Empty diagram class."""

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        """Returns the bounding box of a diagram."""
        return BoundingBox.empty()

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        """Returns a list of primitives."""
        return []

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Converts to SVG image."""
        return dwg.g()


@dataclass
class Compose(Diagram):
    """Compose class."""

    box: BoundingBox
    diagram1: Diagram
    diagram2: Diagram

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        """Returns the bounding box of a diagram."""
        return self.box.apply_transform(t)

    def get_subdiagram_bounding_box(
        self, name: str, t: tx.Transform = Ident
    ) -> Optional[BoundingBox]:
        """Get the bounding box of the sub-diagram."""
        bb = self.diagram1.get_subdiagram_bounding_box(name, t)
        if bb is None:
            bb = self.diagram2.get_subdiagram_bounding_box(name, t)
        return bb

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        """Returns a list of primitives."""
        return self.diagram1.to_list(t) + self.diagram2.to_list(t)

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Converts to SVG image."""
        g = dwg.g()
        g.add(self.diagram1.to_svg(dwg, style))
        g.add(self.diagram2.to_svg(dwg, style))
        return g


@dataclass
class ApplyTransform(Diagram):
    """ApplyTransform class."""

    transform: tx.Transform
    diagram: Diagram

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        """Returns the bounding box of a diagram."""
        t_new = tx.Compose(t, self.transform)
        return self.diagram.get_bounding_box(t_new)

    def get_subdiagram_bounding_box(
        self, name: str, t: tx.Transform = Ident
    ) -> Optional[BoundingBox]:
        """Get the bounding box of the sub-diagram."""
        t_new = tx.Compose(t, self.transform)
        return self.diagram.get_subdiagram_bounding_box(name, t_new)

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        """Returns a list of primitives."""
        t_new = tx.Compose(t, self.transform)
        return [
            prim.apply_transform(t_new) for prim in self.diagram.to_list(t)
        ]

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Converts to SVG image."""
        g = dwg.g(transform=self.transform.to_svg())
        g.add(self.diagram.to_svg(dwg, style))
        return g


@dataclass
class ApplyStyle(Diagram):
    """ApplyStyle class."""

    style: Style
    diagram: Diagram

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        """Returns the bounding box of a diagram."""
        return self.diagram.get_bounding_box(t)

    def get_subdiagram_bounding_box(
        self, name: str, t: tx.Transform = Ident
    ) -> Optional[BoundingBox]:
        """Get the bounding box of the sub-diagram."""
        return self.diagram.get_subdiagram_bounding_box(name, t)

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        """Returns a list of primitives."""
        return [
            prim.apply_style(self.style) for prim in self.diagram.to_list(t)
        ]

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Converts to SVG image."""
        return self.diagram.to_svg(dwg, self.style.merge(style))


@dataclass
class ApplyName(Diagram):
    """ApplyName class."""

    dname: str
    diagram: Diagram

    def get_bounding_box(self, t: tx.Transform = Ident) -> BoundingBox:
        """Returns the bounding box of a diagram."""
        return self.diagram.get_bounding_box(t)

    def get_subdiagram_bounding_box(
        self, name: str, t: tx.Transform = Ident
    ) -> Optional[BoundingBox]:
        """Get the bounding box of the sub-diagram."""
        if name == self.dname:
            return self.diagram.get_bounding_box(t)
        else:
            return None

    def to_list(self, t: tx.Transform = Ident) -> List["Primitive"]:
        """Returns a list of primitives."""
        return [prim for prim in self.diagram.to_list(t)]

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Converts to SVG image."""
        g = dwg.g()
        g.add(self.diagram.to_svg(dwg, style))
        return g
