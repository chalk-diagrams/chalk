from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Tuple

from svgwrite import Drawing
from svgwrite.base import BaseElement

import chalk.transform as tx
from chalk.envelope import Envelope
from chalk.style import StylableProtocol, Style
from chalk.trace import Trace
from chalk.transform import P2, V2, Transformable

if TYPE_CHECKING:
    from chalk.shape import Shape
    from chalk.visitor import A, DiagramVisitor

__all__ = ["BaseElement", "Drawing"]
PyLatexElement = Any
PyLatex = Any
PyCairoContext = Any
PyCairoSurface = Any
Ident = tx.Affine.identity()


class Enveloped(Protocol):
    def get_envelope(self, t: tx.Affine = Ident) -> Envelope:
        ...


class Traceable(Protocol):
    def get_trace(self, t: tx.Affine = Ident) -> Trace:
        ...


class SegmentLike(Enveloped, Traceable, Transformable):
    p: P2
    q: P2
    def render_svg_path(self) -> str: ...
    def render_path(self, ctx: PyCairoContext) -> str: ...
    def render_tikz_path(self, a:Any, b:Any) -> str: ...

class Diagram(
    Enveloped, Traceable, StylableProtocol, tx.TransformableProtocol
):
    def apply_transform(self, t: tx.Affine) -> Diagram:
        ...

    def __add__(self: Diagram, other: Diagram) -> Diagram:
        ...

    def __or__(self, d: Diagram) -> Diagram:
        ...

    def __truediv__(self, d: Diagram) -> Diagram:
        ...

    def __floordiv__(self, d: Diagram) -> Diagram:
        ...

    def juxtapose_snug(
        self: Diagram, other: Diagram, direction: V2
    ) -> Diagram:
        ...

    def beside_snug(self: Diagram, other: Diagram, direction: V2) -> Diagram:
        ...

    def juxtapose(self: Diagram, other: Diagram, direction: V2) -> Diagram:
        ...

    def atop(self: Diagram, other: Diagram) -> Diagram:
        ...

    def above(self: Diagram, other: Diagram) -> Diagram:
        ...

    def beside(self: Diagram, other: Diagram, direction: V2) -> Diagram:
        ...

    def frame(self, extra: float) -> Diagram:
        ...

    def pad(self, extra: float) -> Diagram:
        ...

    def scale_uniform_to_x(self, x: float) -> Diagram:
        ...

    def scale_uniform_to_y(self, y: float) -> Diagram:
        ...

    def align(self: Diagram, v: V2) -> Diagram:
        ...

    def align_t(self: Diagram) -> Diagram:
        ...

    def align_b(self: Diagram) -> Diagram:
        ...

    def align_l(self: Diagram) -> Diagram:
        ...

    def align_r(self: Diagram) -> Diagram:
        ...

    def align_tl(self: Diagram) -> Diagram:
        ...

    def align_tr(self: Diagram) -> Diagram:
        ...

    def align_bl(self: Diagram) -> Diagram:
        ...

    def align_br(self: Diagram) -> Diagram:
        ...

    def snug(self: Diagram, v: V2) -> Diagram:
        ...

    def center_xy(self: Diagram) -> Diagram:
        ...

    def get_subdiagram(
        self, name: str, t: tx.Affine = Ident
    ) -> Optional[Tuple[Diagram, tx.Affine]]:
        ...

    def get_subdiagram_trace(self, name: str, t: tx.Affine = Ident) -> Trace:
        ...

    def get_subdiagram_envelope(
        self, name: str, t: tx.Affine = Ident
    ) -> Envelope:
        ...

    def to_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        ...

    def to_tikz(self, pylatex: PyLatex, style: Style) -> List[PyLatexElement]:
        ...

    def _style(self, style: Style) -> Diagram:
        ...

    def with_envelope(self, other: Diagram) -> Diagram:
        ...

    def show_origin(self) -> Diagram:
        ...

    def show_envelope(self, phantom: bool = False, angle: int = 45) -> Diagram:
        ...

    def compose(
        self, envelope: Envelope, other: Optional[Diagram] = None
    ) -> Diagram:
        ...

    def to_list(self, t: tx.Affine = Ident) -> List[Diagram]:
        ...

    @staticmethod
    def from_shape(shape: Shape) -> Diagram:
        ...

    def accept(self, visitor: DiagramVisitor[A], **kwargs: Any) -> A:
        ...
