from __future__ import annotations

import os
import tempfile

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TypeVar

from svgwrite import Drawing
from svgwrite.base import BaseElement

import chalk.align
import chalk.arrows
import chalk.debug
import chalk.juxtapose
import chalk.padding
import chalk.subdiagram
import chalk.types

from chalk import transform as tx
from chalk.envelope import Envelope, GetEnvelope
from chalk.shape import Shape
from chalk.style import Stylable, Style
from chalk.subdiagram import Subdiagram, GetSubdiagram
from chalk.trace import Trace, GetTrace
from chalk.transform import Affine, unit_x, unit_y
from chalk.types import Diagram
from chalk.utils import imgen
from chalk.visitor import DiagramVisitor

from chalk.backend.cairo import render as render_cairo
from chalk.backend.svg import render as render_svg, ToSVG
from chalk.backend.tikz import render as render_tikz


Trail = Any
Ident = Affine.identity()
A = TypeVar("A")
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
    render = render_cairo
    render_svg = render_svg
    render_pdf = render_tikz

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        return self.accept(ToSVG(), dwg=dwg, style=style)

    def _repr_svg_(self) -> str:
        global SVG_HEIGHT
        f = tempfile.NamedTemporaryFile(delete=False)
        self.render_svg(f.name, height=SVG_HEIGHT)
        f.close()
        svg = open(f.name).read()
        os.unlink(f.name)
        return svg

    # Getters
    def get_envelope(self, t: Affine = Ident) -> Envelope:
        return self.accept(GetEnvelope(), t)

    def get_trace(self, t: Affine = Ident) -> Trace:
        return self.accept(GetTrace(), t)

    def get_subdiagram(
        self, name: str, t: Affine = Ident
    ) -> Optional[Subdiagram]:
        return self.accept(GetSubdiagram(), name, t)

    get_subdiagram_envelope = chalk.subdiagram.get_subdiagram_envelope
    get_subdiagram_trace = chalk.subdiagram.get_subdiagram_trace

    def accept(self, visitor: DiagramVisitor[A], *args: Any, **kwargs: Any) -> A:
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

    def accept(self, visitor: DiagramVisitor[A], *args: Any, **kwargs: Any) -> A:
        return visitor.visit_primitive(self, *args, **kwargs)


@dataclass
class Empty(BaseDiagram):
    """An Empty diagram class."""

    def accept(self, visitor: DiagramVisitor[A], *args: Any, **kwargs: Any) -> A:
        return visitor.visit_empty(self, *args, **kwargs)


@dataclass
class Compose(BaseDiagram):
    """Compose class."""

    envelope: Envelope
    diagram1: BaseDiagram
    diagram2: BaseDiagram

    def accept(self, visitor: DiagramVisitor[A], *args: Any, **kwargs: Any) -> A:
        return visitor.visit_compose(self, *args, **kwargs)


@dataclass
class ApplyTransform(BaseDiagram):
    """ApplyTransform class."""

    transform: Affine
    diagram: BaseDiagram

    def accept(self, visitor: DiagramVisitor[A], *args: Any, **kwargs: Any) -> A:
        return visitor.visit_apply_transform(self, *args, **kwargs)


@dataclass
class ApplyStyle(BaseDiagram):
    """ApplyStyle class."""

    style: Style
    diagram: BaseDiagram

    def accept(self, visitor: DiagramVisitor[A], *args: Any, **kwargs: Any) -> A:
        return visitor.visit_apply_style(self, *args, **kwargs)


@dataclass
class ApplyName(BaseDiagram):
    """ApplyName class."""

    dname: str
    diagram: BaseDiagram

    def accept(self, visitor: DiagramVisitor[A], *args: Any, **kwargs: Any) -> A:
        return visitor.visit_apply_name(self, *args, **kwargs)
