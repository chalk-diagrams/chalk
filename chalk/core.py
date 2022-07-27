from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import chalk.align
import chalk.arrow
import chalk.combinators
import chalk.model
import chalk.types
from chalk import transform as tx
from chalk.envelope import Envelope
from chalk.shape import Shape
from chalk.style import Stylable, Style
from chalk.trace import Trace
from chalk.transform import Affine, unit_x, unit_y
from chalk.types import Diagram
from chalk.utils import imgen

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

    def accept(self, visitor, *args, **kwargs):
        raise NotImplementedError

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
    from chalk.backend.cairo import render as render_cairo
    from chalk.backend.svg import render as render_svg
    from chalk.backend.tikz import render as render_tikz

    render = render_cairo
    render_svg = render_svg
    render_pdf = render_tikz

    def _repr_svg_(self) -> str:
        global SVG_HEIGHT
        f = tempfile.NamedTemporaryFile(delete=False)
        self.render_svg(f.name, height=SVG_HEIGHT)
        f.close()
        svg = open(f.name).read()
        os.unlink(f.name)
        return svg

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

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_primitive(self, *args, **kwargs)


@dataclass
class Empty(BaseDiagram):
    """An Empty diagram class."""

    def get_envelope(self, t: Affine = Ident) -> Envelope:
        """Returns the bounding envelope of a diagram."""
        return Envelope.empty()

    def get_trace(self, t: Affine = Ident) -> Trace:
        return Trace.empty()

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_empty(self, *args, **kwargs)


def empty() -> Diagram:
    return Empty()


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

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_compose(self, *args, **kwargs)


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

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_apply_transform(self, *args, **kwargs)


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

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_apply_style(self, *args, **kwargs)


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

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit_apply_name(self, *args, **kwargs)
