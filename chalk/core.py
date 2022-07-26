from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import svgwrite
from svgwrite import Drawing
from svgwrite.base import BaseElement

import chalk.align
import chalk.arrows
import chalk.debug
import chalk.juxtapose
import chalk.padding
import chalk.types
from chalk import transform as tx
from chalk.envelope import Envelope
from chalk.shape import Shape, Spacer, render_cairo_prims
from chalk.style import Stylable, Style
from chalk.trace import Trace
from chalk.transform import Affine, unit_x, unit_y
from chalk.types import Diagram
from chalk.utils import imgen

PyCairoContext = Any
PyLatex = Any
PyLatexElement = Any
Trail = Any
Ident = Affine.identity()
SVG_HEIGHT = 200


def set_svg_height(height: int) -> None:
    "Globally set the svg preview height for notebooks."
    global SVG_HEIGHT
    SVG_HEIGHT = height


@dataclass
class BaseDiagram(Stylable, tx.Transformable, chalk.types.Diagram):
    """Diagram class."""

    # Core composition
    def apply_transform(self, t: Affine) -> Diagram:  # type: ignore
        return ApplyTransform(t, self)

    def apply_style(self, style: Style) -> Diagram:  # type: ignore
        return ApplyStyle(style, self)

    def _style(self, style: Style) -> Diagram:
        return self.apply_style(style)

    def compose(
        self, envelope: Envelope, other: Optional[Diagram] = None
    ) -> Diagram:
        return Compose(envelope, self, other if other is not None else Empty())

    def named(self, name: str) -> Diagram:
        """Add a name to a diagram.

        Args:
            name (str): Diagram name.

        Returns:
            Diagram: A diagram object.
        """
        return ApplyName(name, self)

    # Juxtapose
    juxtapose_snug = chalk.juxtapose.juxtapose_snug
    beside_snug = chalk.juxtapose.beside_snug
    above = chalk.juxtapose.above
    atop = chalk.juxtapose.atop
    beside = chalk.juxtapose.beside
    above = chalk.juxtapose.above

    # Align
    align = chalk.align.align
    align_t = chalk.align.align_t
    align_b = chalk.align.align_b
    align_l = chalk.align.align_l
    align_r = chalk.align.align_r
    align_tr = chalk.align.align_tr
    align_tl = chalk.align.align_tl
    align_bl = chalk.align.align_bl
    align_br = chalk.align.align_br
    center_xy = chalk.align.center_xy
    with_envelope = chalk.align.with_envelope

    # Arrows
    connect = chalk.arrows.connect
    connect_outside = chalk.arrows.connect_outside
    connect_perim = chalk.arrows.connect_perim

    # Debug
    show_origin = chalk.debug.show_origin
    show_envelope = chalk.debug.show_envelope
    show_beside = chalk.debug.show_beside

    # Padding
    frame = chalk.padding.frame
    pad = chalk.padding.pad
    scale_uniform_to_y = chalk.padding.scale_uniform_to_y
    scale_uniform_to_x = chalk.padding.scale_uniform_to_x

    # Infix
    def __or__(self, d: Diagram) -> Diagram:
        return chalk.juxtapose.beside(self, d, unit_x)

    __truediv__ = chalk.juxtapose.above
    __floordiv__ = chalk.juxtapose.above2
    __add__ = chalk.juxtapose.atop

    # Core rendering (factor out later)
    def to_list(self, t: Affine = Ident) -> List[Diagram]:
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
        import cairo

        envelope = self.get_envelope()
        assert envelope is not None

        pad = 0.05

        # infer width to preserve aspect ratio
        width = width or int(height * envelope.width / envelope.height)

        # determine scale to fit the largest axis in the target frame size
        if envelope.width - width <= envelope.height - height:
            α = height / ((1 + pad) * envelope.height)
        else:
            α = width / ((1 + pad) * envelope.width)

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        s = self.scale(α).center_xy().pad(1 + pad)
        e = s.get_envelope()
        assert e is not None
        s = s.translate(e(-unit_x), e(-unit_y))
        render_cairo_prims(s, ctx, Style.root(max(width, height)))
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

        outer = dwg.g(
            style="fill:white;",
        )
        # Arrow marker
        marker = dwg.marker(
            id="arrow", refX=5.0, refY=1.7, size=(5, 3.5), orient="auto"
        )
        marker.add(dwg.polygon([(0, 0), (5, 1.75), (0, 3.5)]))
        dwg.defs.add(marker)

        dwg.add(outer)
        s = self.center_xy().pad(1 + pad).scale(α)
        e = s.get_envelope()
        assert e is not None
        s = s.translate(e(-unit_x), e(-unit_y))
        style = Style.root(output_size=max(height, width))
        outer.add(s.to_svg(dwg, style))
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
        diagram = self.scale(α).reflect_y().pad(1 + pad)
        envelope = diagram.get_envelope()
        assert envelope is not None
        padding = Primitive.from_shape(
            Spacer(envelope.width, envelope.height)
        ).translate(envelope.center.x, envelope.center.y)
        diagram = diagram + padding
        with doc.create(pylatex.TikZ()) as pic:
            for x in diagram.to_tikz(pylatex, Style.root(max(height, width))):
                pic.append(x)
        doc.generate_tex(path.replace(".pdf", "") + ".tex")
        doc.generate_pdf(path.replace(".pdf", ""), clean_tex=False)

    def _repr_svg_(self) -> str:
        global SVG_HEIGHT
        f = tempfile.NamedTemporaryFile(delete=False)
        self.render_svg(f.name, height=SVG_HEIGHT)
        f.close()
        svg = open(f.name).read()
        os.unlink(f.name)
        return svg

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        """Convert a diagram to SVG image."""
        raise NotImplementedError

    def to_tikz(self, pylatex: PyLatex, style: Style) -> List[PyLatexElement]:
        """Convert a diagram to SVG image."""
        raise NotImplementedError

    # Getters
    def get_subdiagram_envelope(
        self, name: str, t: Affine = Ident
    ) -> Envelope:
        """Get the bounding envelope of the sub-diagram."""
        subdiagram = self.get_subdiagram(name)
        assert subdiagram is not None, "Subdiagram does not exist"
        return subdiagram[0].get_envelope(subdiagram[1])

    def get_subdiagram_trace(self, name: str, t: Affine = Ident) -> Trace:
        """Get the trace of the sub-diagram."""
        subdiagram = self.get_subdiagram(name)
        assert subdiagram is not None, "Subdiagram does not exist"
        return subdiagram[0].get_trace(subdiagram[1])

    def get_subdiagram(
        self, name: str, t: Affine = Ident
    ) -> Optional[Tuple[Diagram, Affine]]:
        return None


@dataclass
class Primitive(BaseDiagram):
    """Primitive class.

    This is derived from a ``chalk.core.Diagram`` class.

    [TODO]: explain what Primitive class is for.
    """

    shape: Shape
    style: Style
    transform: Affine

    @classmethod
    def from_shape(cls, shape: Shape) -> Diagram:
        """Create and return a primitive from a shape.

        Args:
            shape (Shape): A shape object.

        Returns:
            Diagram: A diagram object.
        """
        return cls(shape, Style.empty(), Ident)

    def apply_transform(self, t: Affine) -> Diagram:  # type: ignore
        """Applies a transform and returns a primitive.

        Args:
            t (Transform): A transform object.

        Returns:
            Diagram
        """
        new_transform = t * self.transform
        return Primitive(self.shape, self.style, new_transform)

    def apply_style(self, other_style: Style) -> Primitive:
        """Applies a style and returns a primitive.

        Args:
            other_style (Style): A style object.

        Returns:
            Diagram
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
        return self.shape.get_envelope().apply_transform(new_transform)

    def get_trace(self, t: Affine = Ident) -> Trace:
        new_transform = t * self.transform
        return self.shape.get_trace().apply_transform(new_transform)

    def to_list(self, t: Affine = Ident) -> List[Diagram]:
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
        style = self.style.merge(other_style)
        style_svg = style.to_svg()
        transform = tx.to_svg(self.transform)
        inner = self.shape.render_svg(dwg, style)
        if not style_svg and not transform:
            return inner
        else:
            if not style_svg:
                style_svg = ";"
            g = dwg.g(transform=transform, style=style_svg)
            g.add(inner)
            return g

    def to_tikz(
        self, pylatex: PyLatexElement, other_style: Style
    ) -> List[PyLatexElement]:
        """Convert a diagram to SVG image."""

        transform = tx.to_tikz(self.transform)
        style = self.style.merge(other_style)
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
class Empty(BaseDiagram):
    """An Empty diagram class."""

    def get_envelope(self, t: Affine = Ident) -> Envelope:
        """Returns the bounding envelope of a diagram."""
        return Envelope.empty()

    def get_trace(self, t: Affine = Ident) -> Trace:
        return Trace.empty()

    def to_list(self, t: Affine = Ident) -> List[Diagram]:
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
class Compose(BaseDiagram):
    """Compose class."""

    envelope: Envelope
    diagram1: Diagram
    diagram2: Diagram

    def get_envelope(self, t: Affine = Ident) -> Envelope:
        """Returns the bounding envelope of a diagram."""
        return self.envelope.apply_transform(t)

    def get_trace(self, t: Affine = Ident) -> Trace:
        # TODO Should we cache the trace?
        return self.diagram1.get_trace(t) + self.diagram2.get_trace(t)

    def get_subdiagram(
        self, name: str, t: Affine = Ident
    ) -> Optional[Tuple[Diagram, Affine]]:
        """Get the bounding envelope of the sub-diagram."""
        bb = self.diagram1.get_subdiagram(name, t)
        if bb is None:
            bb = self.diagram2.get_subdiagram(name, t)
        return bb

    def to_list(self, t: Affine = Ident) -> List[Diagram]:
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
class ApplyTransform(BaseDiagram):
    """ApplyTransform class."""

    transform: Affine
    diagram: Diagram

    def get_envelope(self, t: Affine = Ident) -> Envelope:
        """Returns the bounding envelope of a diagram."""
        n = t * self.transform
        return self.diagram.get_envelope(n)

    def get_trace(self, t: Affine = Ident) -> Trace:
        """Returns the bounding envelope of a diagram."""
        return self.diagram.get_trace(t * self.transform)

    def get_subdiagram(
        self, name: str, t: Affine = Ident
    ) -> Optional[Tuple[Diagram, Affine]]:
        """Get the bounding envelope of the sub-diagram."""
        return self.diagram.get_subdiagram(name, t * self.transform)

    def to_list(self, t: Affine = Ident) -> List[Diagram]:
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
        options["cm"] = tx.to_tikz(self.transform)
        s = pylatex.TikZScope(options=pylatex.TikZOptions(**options))
        for x in self.diagram.to_tikz(pylatex, style):
            s.append(x)
        return [s]


@dataclass
class ApplyStyle(BaseDiagram):
    """ApplyStyle class."""

    style: Style
    diagram: Diagram

    def get_envelope(self, t: Affine = Ident) -> Envelope:
        """Returns the bounding envelope of a diagram."""
        return self.diagram.get_envelope(t)

    def get_trace(self, t: Affine = Ident) -> Trace:
        """Returns the bounding envelope of a diagram."""
        return self.diagram.get_trace(t)

    def get_subdiagram(
        self, name: str, t: Affine = Ident
    ) -> Optional[Tuple[Diagram, Affine]]:
        return self.diagram.get_subdiagram(name, t)

    def to_list(self, t: Affine = Ident) -> List[Diagram]:
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
class ApplyName(BaseDiagram):
    """ApplyName class."""

    dname: str
    diagram: Diagram

    def get_envelope(self, t: Affine = Ident) -> Envelope:
        """Returns the bounding envelope of a diagram."""
        return self.diagram.get_envelope(t)

    def get_trace(self, t: Affine = Ident) -> Trace:
        """Returns the bounding envelope of a diagram."""
        return self.diagram.get_trace(t)

    def get_subdiagram(
        self, name: str, t: Affine = Ident
    ) -> Optional[Tuple[Diagram, Affine]]:
        """Get the bounding envelope of the sub-diagram."""
        if name == self.dname:
            return self.diagram, t
        else:
            return None

    def to_list(self, t: Affine = Ident) -> List[Diagram]:
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
