from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cairo
import svgwrite
from colour import Color
from planar import Affine, Vec2
from svgwrite import Drawing
from svgwrite.base import BaseElement

from chalk import transform as tx
from chalk.envelope import Envelope
from chalk.shape import Circle, Path, Shape, Spacer
from chalk.style import Style
from chalk.trace import Trace
from chalk.utils import imgen

PyCairoContext = Any
PyLatex = Any
PyLatexElement = Any
Ident = Affine.identity()
unit_x = Vec2(1, 0)
unit_y = Vec2(0, 1)


@dataclass
class Diagram(tx.Transformable):
    """Diagram class."""

    def get_envelope(self, t: Affine = Ident) -> Optional[Envelope]:
        """Get the envelope of a diagram."""
        raise NotImplementedError

    def get_trace(self, t: Affine = Ident) -> Trace:
        """Get the trace of a diagram."""
        raise NotImplementedError

    def to_list(self, t: Affine = Ident) -> List["Primitive"]:
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
        envelope = self.get_envelope()
        assert envelope is not None

        # infer width to preserve aspect ratio
        width = width or int(height * envelope.width / envelope.height)

        # determine scale to fit the largest axis in the target frame size
        if envelope.width - width <= envelope.height - height:
            α = height / ((1 + pad) * envelope.height)
        else:
            α = width / ((1 + pad) * envelope.width)

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        ctx.scale(α, α)

        tl = envelope.min_point
        ctx.translate(-(1 + pad) * tl.x, -(1 + pad) * tl.y)

        prims = self.to_list()

        for prim in prims:
            # apply transformation
            matrix = tx.to_cairo(prim.transform)
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
        envelope = self.get_envelope()

        # infer width to preserve aspect ratio
        assert envelope is not None
        width = width or int(height * envelope.width / envelope.height)

        # determine scale to fit the largest axis in the target frame size
        if envelope.width - width <= envelope.height - height:
            α = height / ((1 + pad) * envelope.height)
        else:
            α = width / ((1 + pad) * envelope.width)
        dwg = svgwrite.Drawing(
            path,
            size=(width, height),
        )
        tl = envelope.min_point
        x, y = -(1 + pad) * tl.x, -(1 + pad) * tl.y
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

    def render_pdf(self, path: str, height: int = 128) -> None:
        # Hack: Convert roughly from px to pt. Assume 300 dpi.
        heightpt = height / 4.3
        try:
            import pylatex
        except ImportError:
            print("Render PDF requires pylatex installation.")
            return

        pad = 0.05
        envelope = self.get_envelope()
        assert envelope is not None

        # infer width to preserve aspect ratio
        width = heightpt * (envelope.width / envelope.height)
        # determine scale to fit the largest axis in the target frame size
        if envelope.width - width <= envelope.height - heightpt:
            α = heightpt / ((1 + pad) * envelope.height)
        else:
            α = width / ((1 + pad) * envelope.width)
        x, _ = pad * heightpt, pad * width

        # create document
        doc = pylatex.Document(documentclass="standalone")
        # document_options= pylatex.TikZOptions(margin=f"{{{x}pt {x}pt {y}pt {y}pt}}"))
        # add our sample drawings
        diagram = self.scale(α).reflect_y().frame(x)
        envelope = diagram.get_envelope()
        assert envelope is not None
        padding = Primitive.from_shape(
            Spacer(envelope.width, envelope.height)
        ).translate(envelope.center.x, envelope.center.y)
        diagram = diagram + padding
        with doc.create(pylatex.TikZ()) as pic:
            for x in diagram.to_tikz(pylatex, Style.default()):
                pic.append(x)
        doc.generate_tex(path.replace(".pdf", "") + ".tex")
        doc.generate_pdf(path.replace(".pdf", ""), clean_tex=False)

    def _repr_svg_(self) -> str:
        f = tempfile.NamedTemporaryFile(delete=False)
        self.render_svg(f.name)
        f.close()
        svg = open(f.name).read()
        os.unlink(f.name)
        return svg

    def _empty_check(
        self, other: Diagram
    ) -> Tuple[Optional[Diagram], Envelope, Envelope]:
        fake = Envelope(lambda v: 0.0)
        envelope1 = self.get_envelope()
        envelope2 = other.get_envelope()
        if envelope2 is None:
            return self, fake, fake
        if envelope1 is None:
            return other, fake, fake
        return None, envelope1, envelope2

    def atop(
        self, other: Diagram, direction: Optional[Vec2] = None
    ) -> Diagram:
        dia, envelope1, envelope2 = self._empty_check(other)
        if dia is not None:
            return dia
        if direction is None:
            new_envelope = envelope1 + envelope2
            return Compose(new_envelope, self, other)
        else:
            d = envelope1(direction) + envelope2(-direction)
            t = Affine.translation(direction * d)
            new_envelope = envelope1 + (t * envelope2)
            return Compose(new_envelope, self, ApplyTransform(t, other))

    __add__ = atop

    def above(self, other: Diagram) -> Diagram:
        return self.atop(other, unit_y)

    __truediv__ = above

    def beside(self, other: Diagram) -> Diagram:
        return self.atop(other, unit_x)

    __or__ = beside

    def above2(self, other: Diagram) -> Diagram:
        """Given two diagrams ``a`` and ``b``, ``a.above2(b)``
        places ``a`` on top of ``b``. This moves ``a`` down to
        touch ``b``.

        💡 ``a.above2(b)`` is equivalent to ``a // b``.

        Args:
            other (Diagram): Another diagram object.

        Returns:
            Diagram: A diagram object.
        """
        dia, envelope1, envelope2 = self._empty_check(other)
        if dia is not None:
            return dia
        direction = unit_y
        d = envelope1(direction) + envelope2(-direction)
        t = Affine.translation(-direction * d)
        new_envelope = t * envelope1 + envelope2
        return Compose(new_envelope, ApplyTransform(t, self), other)

    __floordiv__ = above2

    def center_xy(self) -> Diagram:
        """Center< a diagram.

        Returns:
            Diagram: A diagram object.
        """
        envelope = self.get_envelope()
        if envelope is None:
            return self
        t = Affine.translation(-envelope.center)
        return ApplyTransform(t, self)

    def align(self, v: Vec2) -> Diagram:
        envelope = self.get_envelope()
        if envelope is None:
            return self

        t = Affine.translation(-envelope.envelope_v(v))
        return ApplyTransform(t, self)

    def align_t(self) -> Diagram:
        """Align a diagram with its top edge.

        Returns:
            Diagram
        """
        return self.align(-unit_y)

    def align_b(self) -> Diagram:
        """Align a diagram with its bottom edge.

        Returns:
            Diagram
        """
        return self.align(unit_y)

    def align_r(self) -> Diagram:
        """Align a diagram with its right edge.

        Returns:
            Diagram
        """
        return self.align(unit_x)

    def align_l(self) -> Diagram:
        """Align a diagram with its left edge.

        Returns:
            Diagram: A diagram object.
        """
        return self.align(-unit_x)

    def align_tl(self) -> Diagram:
        """Align a diagram with its top-left edges.

        Returns:
            Diagram
        """
        return self.align_t().align_l()

    def align_br(self) -> Diagram:
        """Align a diagram with its bottom-right edges.

        Returns:
            Diagram: A diagram object.
        """
        return self.align_b().align_r()

    def align_tr(self) -> Diagram:
        """Align a diagram with its top-right edges.

        Returns:
            Diagram: A diagram object.
        """
        return self.align_t().align_r()

    def align_bl(self) -> Diagram:
        """Align a diagram with its bottom-left edges.

        Returns:
            Diagram: A diagram object.
        """
        return self.align_b().align_l()

    # def pad_l(self, extra: float) -> Diagram:
    #     """Add outward directed left-side padding for
    #     a diagram. This padding is applied **only** on
    #     the **left** side.

    #     Args:
    #         extra (float): Amount of padding to add.

    #     Returns:
    #         Diagram: A diagram object.
    #     """
    #     return self
    #     envelope = self.get_envelope()
    #     if envelope is None:
    #         return self
    #     tl, br = envelope.min_point, envelope.max_point
    #     new_envelope = Envelope.from_points([Point(tl.x - extra, tl.y), br])
    #     return Compose(new_envelope, self, Empty())

    # def pad_t(self, extra: float) -> Diagram:
    #     """Add outward directed top-side padding for
    #     a diagram. This padding is applied **only** on
    #     the **top** side.

    #     Args:
    #         extra (float): Amount of padding to add.

    #     Returns:
    #         Diagram: A diagram object.
    #     """
    #     return self
    #     envelope = self.get_envelope()
    #     if envelope is None:
    #         return self
    #     tl, br = envelope.min_point, envelope.max_point
    #     new_envelope = Envelope.from_points([Point(tl.x, tl.y - extra), br])
    #     return Compose(new_envelope, self, Empty())

    # def pad_r(self, extra: float) -> Diagram:
    #     """Add outward directed right-side padding for
    #     a diagram. This padding is applied **only** on
    #     the **right** side.

    #     Args:
    #         extra (float): Amount of padding to add.

    #     Returns:
    #         Diagram: A diagram object.
    #     """
    #     return self
    #     envelope = self.get_envelope()
    #     if envelope is None:
    #         return self
    #     tl, br = envelope.min_point, envelope.max_point
    #     new_envelope = Envelope.from_points([tl, Point(br.x + extra, br.y)])
    #     return Compose(new_envelope, self, Empty())

    # def pad_b(self, extra: float) -> Diagram:
    #     """Add outward directed bottom-side padding for
    #     a diagram. This padding is applied **only** on
    #     the **bottom** side.

    #     Args:
    #         extra (float): Amount of padding to add.

    #     Returns:
    #         Diagram: A diagram object.
    #     """
    #     return self
    #     envelope = self.get_envelope()
    #     if envelope is None:
    #         return self
    #     tl, br = envelope.min_point, envelope.max_point
    #     new_envelope = Envelope.from_points([tl, Point(br.x, br.y + extra)])
    #     return Compose(new_envelope, self, Empty())

    def frame(self, extra: float) -> Diagram:
        """Add outward directed padding for a diagram.
        This padding is applied uniformly on all sides.

        Args:
            extra (float): Amount of padding to add.

        Returns:
            Diagram: A diagram object.
        """
        envelope = self.get_envelope()
        if envelope is None:
            return self

        def f(d: Vec2) -> float:
            assert envelope is not None
            return envelope(d) + extra

        new_envelope = Envelope(f)
        return Compose(new_envelope, self, Empty())

    def scale_uniform_to_x(self, x: float) -> Diagram:
        """Apply uniform scaling along the x-axis.

        Args:
            x (float): Amount of scaling along the x-axis.

        Returns:
            Diagram: A diagram object.
        """
        envelope = self.get_envelope()
        if envelope is None:
            return self
        α = x / envelope.width
        return ApplyTransform(Affine.scale(Vec2(α, α)), self)

    def scale_uniform_to_y(self, y: float) -> Diagram:
        """Apply uniform scaling along the y-axis.

        Args:
            y (float): Amount of scaling along the y-axis.

        Returns:
            Diagram: A diagram object.
        """
        envelope = self.get_envelope()
        if envelope is None:
            return self
        α = y / envelope.height
        return ApplyTransform(Affine.scale(Vec2(α, α)), self)

    def apply_transform(self, t: Affine) -> Diagram:  # type: ignore
        """Apply a transformation.

        Args:
            t (Affine): A transformation.

        Returns:
            Diagram: A diagram object.
        """
        return ApplyTransform(t, self)

    # def at(self, x: float, y: float) -> Diagram:
    #     t = tx.Translate(x, y)
    #     return ApplyTransform(t, self.center_xy())

    def line_width(self, width: float) -> Diagram:
        """Apply specified line-width to the edge of
        the diagram.

        Args:
            width (float): Amount of width.

        Returns:
            Diagram: A diagram object.
        """
        return ApplyStyle(Style(line_width=width), self)

    def line_color(self, color: Color) -> Diagram:
        """Apply specified line-color to the edge of
        the diagram.

        Args:
            color (float): A color (``colour.Color``).

        Returns:
            Diagram: A diagram object.
        """
        return ApplyStyle(Style(line_color=color), self)

    def fill_color(self, color: Color) -> Diagram:
        """Apply specified fill-color to the diagram.

        Args:
            color (Color): A color object.

        Returns:
            Diagram: A diagram object.
        """
        return ApplyStyle(Style(fill_color=color), self)

    def fill_opacity(self, opacity: float) -> Diagram:
        """Apply specified amount of opacity to the diagram.

        Args:
            opacity (float): Amount of opacity (between 0 and 1).

        Returns:
            Diagram: A diagram object.
        """
        return ApplyStyle(Style(fill_opacity=opacity), self)

    def dashing(self, dashing_strokes: List[float], offset: float) -> Diagram:
        """Apply dashed line to the edge of a diagram.

        > [TODO]: improve args description.

        Args:
            dashing_strokes (List[float]): Dashing strokes
            offset (float): Amount of offset

        Returns:
            Diagram: A diagram object.
        """
        return ApplyStyle(Style(dashing=(dashing_strokes, offset)), self)

    def at_center(self, other: Diagram) -> Diagram:
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
        dia, envelope1, envelope2 = self._empty_check(other)
        if dia is not None:
            return dia
        t = Affine.translation(envelope1.center)
        new_envelope = envelope1 + (t * envelope2)
        return Compose(new_envelope, self, ApplyTransform(t, other))

    def show_origin(self) -> Diagram:
        """Add a red dot at the origin of a diagram for debugging.

        Returns:
            Diagram
        """
        envelope = self.get_envelope()
        if envelope is None:
            return self
        origin_size = min(envelope.height, envelope.width) / 50
        origin = Primitive(
            Circle(origin_size), Style(fill_color=Color("red")), Ident
        )
        return self + origin

    def show_envelope(self) -> Diagram:
        """Add red envelope to diagram for debugging.

        Returns:
            Diagram
        """
        envelope = self.get_envelope()
        if envelope is None:
            return self
        origin = Primitive(
            Path(envelope.to_path()),
            Style(fill_opacity=0, line_color=Color("red")),
            Ident,
        )
        return self + origin

    def named(self, name: str) -> Diagram:
        """Add a name to a diagram.

        Args:
            name (str): Diagram name.

        Returns:
            Diagram: A diagram object.
        """
        return ApplyName(name, self)

    def get_subdiagram_envelope(
        self, name: str, t: Affine = Ident
    ) -> Optional[Envelope]:
        """Get the bounding envelope of the sub-diagram."""
        return None

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Convert a diagram to SVG image."""
        raise NotImplementedError

    def to_tikz(self, pylatex: PyLatex, style: Style) -> List[PyLatexElement]:
        """Convert a diagram to SVG image."""
        raise NotImplementedError


def bounding_envelope_transform(t: Affine, envelope: Envelope) -> Envelope:
    return envelope.apply_transform(t)


@dataclass
class Primitive(Diagram):
    """Primitive class.

    This is derived from a ``chalk.core.Diagram`` class.

    [TODO]: explain what Primitive class is for.
    """

    shape: Shape
    style: Style
    transform: Affine

    @classmethod
    def from_shape(cls, shape: Shape) -> "Primitive":
        """Create and return a primitive from a shape.

        Args:
            shape (Shape): A shape object.

        Returns:
            Primitive: A primitive object.
        """
        return cls(shape, Style.default(), Ident)

    def apply_transform(self, t: Affine) -> "Primitive":  # type: ignore
        """Applies a transform and returns a primitive.

        Args:
            t (Transform): A transform object.

        Returns:
            Primitive: A primitive object.
        """
        new_transform = t * self.transform
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

    def get_envelope(self, t: Affine = Ident) -> Envelope:
        """Apply a transform and return a bounding envelope.

        Args:
            t (Transform): A transform object
                           Defaults to Ident.

        Returns:
            Envelope: A bounding envelope object.
        """

        new_transform = t * self.transform
        return bounding_envelope_transform(
            new_transform, self.shape.get_envelope()
        )

    def get_trace(self, t: Affine = Ident) -> Trace:
        new_transform = t * self.transform
        return self.shape.get_trace().apply_transform(new_transform)

    def to_list(self, t: Affine = Ident) -> List["Primitive"]:
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
        transform = tx.to_svg(self.transform)
        inner = self.shape.render_svg(dwg)
        if not style and not transform:
            return inner
        else:
            if not style:
                style = ";"
            g = dwg.g(transform=transform, style=style)
            g.add(inner)
            return g

    def to_tikz(
        self, pylatex: PyLatexElement, other_style: Style
    ) -> List[PyLatexElement]:
        """Convert a diagram to SVG image."""

        transform = tx.to_tikz(self.transform)
        style = self.style.merge(other_style)
        style = style.scale_style(max(self.transform[0], self.transform[4]))
        inner = self.shape.render_tikz(pylatex, style)
        if not style and not transform:
            return [inner]
        else:
            options = {}
            options["cm"] = tx.to_tikz(self.transform)
            s = pylatex.TikZScope(options=pylatex.TikZOptions(**options))
            s.append(inner)
            return [s]


@dataclass
class Empty(Diagram):
    """An Empty diagram class."""

    def get_envelope(self, t: Affine = Ident) -> Optional[Envelope]:
        """Returns the bounding envelope of a diagram."""
        return None

    def get_trace(self, t: Affine = Ident) -> Trace:
        return Trace.empty()

    def to_list(self, t: Affine = Ident) -> List["Primitive"]:
        """Returns a list of primitives."""
        return []

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Converts to SVG image."""
        return dwg.g()

    def to_tikz(
        self, pylatex: PyLatexElement, style: Style
    ) -> List[PyLatexElement]:
        """Converts to SVG image."""
        return []


@dataclass
class Compose(Diagram):
    """Compose class."""

    envelope: Envelope
    diagram1: Diagram
    diagram2: Diagram

    def get_envelope(self, t: Affine = Ident) -> Optional[Envelope]:
        """Returns the bounding envelope of a diagram."""
        return bounding_envelope_transform(t, self.envelope)

    def get_trace(self, t: Affine = Ident) -> Trace:
        # TODO Should we cache the trace?
        return self.diagram1.get_trace(t) + self.diagram2.get_trace(t)

    def get_subdiagram_envelope(
        self, name: str, t: Affine = Ident
    ) -> Optional[Envelope]:
        """Get the bounding envelope of the sub-diagram."""
        bb = self.diagram1.get_subdiagram_envelope(name, t)
        if bb is None:
            bb = self.diagram2.get_subdiagram_envelope(name, t)
        return bb

    def to_list(self, t: Affine = Ident) -> List["Primitive"]:
        """Returns a list of primitives."""
        return self.diagram1.to_list(t) + self.diagram2.to_list(t)

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Converts to SVG image."""
        g = dwg.g()
        g.add(self.diagram1.to_svg(dwg, style))
        g.add(self.diagram2.to_svg(dwg, style))
        return g

    def to_tikz(
        self, pylatex: PyLatexElement, style: Style
    ) -> List[PyLatexElement]:
        """Converts to tikz image."""
        return self.diagram1.to_tikz(pylatex, style) + self.diagram2.to_tikz(
            pylatex, style
        )


@dataclass
class ApplyTransform(Diagram):
    """ApplyTransform class."""

    transform: Affine
    diagram: Diagram

    def get_envelope(self, t: Affine = Ident) -> Optional[Envelope]:
        """Returns the bounding envelope of a diagram."""
        n = t * self.transform
        return self.diagram.get_envelope(n)

    def get_trace(self, t: Affine = Ident) -> Trace:
        """Returns the bounding envelope of a diagram."""
        return self.diagram.get_trace(t * self.transform)

    def get_subdiagram_envelope(
        self, name: str, t: Affine = Ident
    ) -> Optional[Envelope]:
        """Get the bounding envelope of the sub-diagram."""
        return self.diagram.get_subdiagram_envelope(name, t * self.transform)

    def to_list(self, t: Affine = Ident) -> List["Primitive"]:
        """Returns a list of primitives."""
        t_new = t * self.transform
        return [
            prim.apply_transform(t_new) for prim in self.diagram.to_list(t)
        ]

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Converts to SVG image."""
        g = dwg.g(transform=tx.to_svg(self.transform))
        g.add(self.diagram.to_svg(dwg, style))
        return g

    def to_tikz(
        self, pylatex: PyLatexElement, style: Style
    ) -> List[PyLatexElement]:
        options = {}
        style = style.scale_style(max(self.transform[0], self.transform[4]))
        options["cm"] = tx.to_tikz(self.transform)
        s = pylatex.TikZScope(options=pylatex.TikZOptions(**options))
        for x in self.diagram.to_tikz(pylatex, style):
            s.append(x)
        return [s]


@dataclass
class ApplyStyle(Diagram):
    """ApplyStyle class."""

    style: Style
    diagram: Diagram

    def get_envelope(self, t: Affine = Ident) -> Optional[Envelope]:
        """Returns the bounding envelope of a diagram."""
        return self.diagram.get_envelope(t)

    def get_trace(self, t: Affine = Ident) -> Trace:
        """Returns the bounding envelope of a diagram."""
        return self.diagram.get_trace(t)

    def get_subdiagram_envelope(
        self, name: str, t: Affine = Ident
    ) -> Optional[Envelope]:
        """Get the bounding envelope of the sub-diagram."""
        return self.diagram.get_subdiagram_envelope(name, t)

    def to_list(self, t: Affine = Ident) -> List["Primitive"]:
        """Returns a list of primitives."""
        return [
            prim.apply_style(self.style) for prim in self.diagram.to_list(t)
        ]

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Converts to SVG image."""
        return self.diagram.to_svg(dwg, self.style.merge(style))

    def to_tikz(
        self, pylatex: PyLatexElement, style: Style
    ) -> List[PyLatexElement]:
        return self.diagram.to_tikz(pylatex, self.style.merge(style))


@dataclass
class ApplyName(Diagram):
    """ApplyName class."""

    dname: str
    diagram: Diagram

    def get_envelope(self, t: Affine = Ident) -> Optional[Envelope]:
        """Returns the bounding envelope of a diagram."""
        return self.diagram.get_envelope(t)

    def get_trace(self, t: Affine = Ident) -> Trace:
        """Returns the bounding envelope of a diagram."""
        return self.diagram.get_trace(t)

    def get_subdiagram_envelope(
        self, name: str, t: Affine = Ident
    ) -> Optional[Envelope]:
        """Get the bounding envelope of the sub-diagram."""
        if name == self.dname:
            return self.diagram.get_envelope(t)
        else:
            return None

    def to_list(self, t: Affine = Ident) -> List["Primitive"]:
        """Returns a list of primitives."""
        return [prim for prim in self.diagram.to_list(t)]

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Converts to SVG image."""
        g = dwg.g()
        g.add(self.diagram.to_svg(dwg, style))
        return g

    def to_tikz(
        self, pylatex: PyLatexElement, style: Style
    ) -> List[PyLatexElement]:
        return self.diagram.to_tikz(pylatex, style)
