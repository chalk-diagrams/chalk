from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Optional, TypeVar

import chalk.align
import chalk.arrow
import chalk.backend.cairo
import chalk.backend.svg
import chalk.backend.tikz
import chalk.combinators
import chalk.model
import chalk.padding
import chalk.subdiagram
import chalk.trace
import chalk.types
from chalk import backend
from chalk import transform as tx
from chalk.envelope import Envelope
from chalk.style import Stylable, Style
from chalk.transform import Affine, unit_x
from chalk.types import Diagram, Shape
from chalk.utils import imgen
from chalk.visitor import DiagramVisitor

Trail = Any
Ident = Affine.identity()
A = TypeVar("A")
SVG_HEIGHT = 200
SVG_DRAW_HEIGHT = None


def set_svg_height(height: int) -> None:
    "Globally set the svg preview height for notebooks."
    global SVG_HEIGHT
    SVG_HEIGHT = height


def set_svg_draw_height(height: int) -> None:
    "Globally set the svg preview height for notebooks."
    global SVG_DRAW_HEIGHT
    SVG_DRAW_HEIGHT = height


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

    # Combinators
    with_envelope = chalk.combinators.with_envelope
    juxtapose = chalk.combinators.juxtapose
    juxtapose_snug = chalk.combinators.juxtapose_snug
    beside_snug = chalk.combinators.beside_snug
    above = chalk.combinators.above
    atop = chalk.combinators.atop
    beside = chalk.combinators.beside
    above = chalk.combinators.above

    # Align
    align = chalk.align.align_to
    align_t = chalk.align.align_t
    align_b = chalk.align.align_b
    align_l = chalk.align.align_l
    align_r = chalk.align.align_r
    align_tr = chalk.align.align_tr
    align_tl = chalk.align.align_tl
    align_bl = chalk.align.align_bl
    align_br = chalk.align.align_br
    center_xy = chalk.align.center_xy
    center = chalk.align.center
    scale_uniform_to_y = chalk.align.scale_uniform_to_y
    scale_uniform_to_x = chalk.align.scale_uniform_to_x

    # Arrows
    connect = chalk.arrow.connect
    connect_outside = chalk.arrow.connect_outside
    connect_perim = chalk.arrow.connect_perim

    # Model
    show_origin = chalk.model.show_origin
    show_envelope = chalk.model.show_envelope
    show_beside = chalk.model.show_beside

    # Combinators
    frame = chalk.combinators.frame
    pad = chalk.combinators.pad

    # Infix
    def __or__(self, d: Diagram) -> Diagram:
        return chalk.combinators.beside(self, d, unit_x)

    __truediv__ = chalk.combinators.above
    __floordiv__ = chalk.combinators.above2
    __add__ = chalk.combinators.atop

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

    # Rendering
    render = chalk.backend.cairo.render
    render_png = chalk.backend.cairo.render
    render_svg = chalk.backend.svg.render
    render_pdf = chalk.backend.tikz.render

    to_svg = backend.svg.to_svg
    to_tikz = backend.tikz.to_tikz

    def _repr_svg_(self) -> str:
        global SVG_HEIGHT
        f = tempfile.NamedTemporaryFile(delete=False)
        self.render_svg(f.name, height=SVG_HEIGHT, draw_height=SVG_DRAW_HEIGHT)
        f.close()
        svg = open(f.name).read()
        os.unlink(f.name)
        return svg

    # Getters
    get_envelope = chalk.envelope.get_envelope
    get_trace = chalk.trace.get_trace

    get_subdiagram = chalk.subdiagram.get_subdiagram
    get_subdiagram_envelope = chalk.subdiagram.get_subdiagram_envelope
    get_subdiagram_trace = chalk.subdiagram.get_subdiagram_trace

    def accept(self, visitor: DiagramVisitor[A], **kwargs: Any) -> A:
        raise NotImplementedError


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
    def from_shape(cls, shape: Shape) -> Primitive:
        """Creates a primitive from a shape using the default style (only line
        stroke, no fill) and the identity transformation.

        Args:
            shape (Shape): A shape object.

        Returns:
            Primitive: A diagram object.
        """
        return cls(shape, Style.empty(), Ident)

    def apply_transform(self, t: Affine) -> Primitive:  # type: ignore
        """Applies a transform and returns a primitive.

        Args:
            t (Transform): A transform object.

        Returns:
            Primitive
        """
        new_transform = t * self.transform
        return Primitive(self.shape, self.style, new_transform)

    def apply_style(self, other_style: Style) -> Primitive:
        """Applies a style and returns a primitive.

        Args:
            other_style (Style): A style object.

        Returns:
            Primitive
        """
        return Primitive(
            self.shape, self.style.merge(other_style), self.transform
        )

    def accept(self, visitor: DiagramVisitor[A], **kwargs: Any) -> A:
        return visitor.visit_primitive(self, **kwargs)


@dataclass
class Empty(BaseDiagram):
    """An Empty diagram class."""

    def accept(self, visitor: DiagramVisitor[A], **kwargs: Any) -> A:
        return visitor.visit_empty(self, **kwargs)


def empty() -> Diagram:
    return Empty()


@dataclass
class Compose(BaseDiagram):
    """Compose class."""

    envelope: Envelope
    diagram1: Diagram
    diagram2: Diagram

    def accept(self, visitor: DiagramVisitor[A], **kwargs: Any) -> A:
        return visitor.visit_compose(self, **kwargs)


@dataclass
class ApplyTransform(BaseDiagram):
    """ApplyTransform class."""

    transform: Affine
    diagram: Diagram

    def accept(self, visitor: DiagramVisitor[A], **kwargs: Any) -> A:
        return visitor.visit_apply_transform(self, **kwargs)


@dataclass
class ApplyStyle(BaseDiagram):
    """ApplyStyle class."""

    style: Style
    diagram: Diagram

    def accept(self, visitor: DiagramVisitor[A], **kwargs: Any) -> A:
        return visitor.visit_apply_style(self, **kwargs)


@dataclass
class ApplyName(BaseDiagram):
    """ApplyName class."""

    dname: str
    diagram: Diagram

    def accept(self, visitor: DiagramVisitor[A], **kwargs: Any) -> A:
        return visitor.visit_apply_name(self, **kwargs)
