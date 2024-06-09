from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol

import chalk.transform as tx
from chalk.envelope import Envelope
from chalk.monoid import Monoid
from chalk.style import Stylable, StyleHolder
from chalk.trace import Trace
from chalk.transform import P2_t, V2_t

if TYPE_CHECKING:
    from chalk.path import Path
    from chalk.subdiagram import Name, Subdiagram
    from chalk.trail import Located, Trail
    from chalk.visitor import A, DiagramVisitor, ShapeVisitor

Ident = tx.ident


class Enveloped(Protocol):
    def get_envelope(self) -> Envelope: ...


class Traceable(Protocol):
    def get_trace(self) -> Trace: ...


class Shape(Enveloped, Traceable, Protocol):
    def accept(self, visitor: ShapeVisitor[A], **kwargs: Any) -> A: ...


class TrailLike(Protocol):
    def to_trail(self) -> Trail: ...

    def to_path(self, location: P2_t = tx.P2(0, 0)) -> Path:
        return self.at(location).to_path()

    def at(self, location: P2_t) -> Located:
        return self.to_trail().at(location)

    def stroke(self) -> Diagram:
        return self.at(tx.P2(0, 0)).stroke()


class Diagram(Enveloped, Traceable, Stylable, tx.Transformable, Monoid):
    def apply_transform(self, t: tx.Affine) -> Diagram:  # type: ignore[empty-body]
        ...

    def __add__(self: Diagram, other: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def __or__(self, d: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def __truediv__(self, d: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def __floordiv__(self, d: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def juxtapose_snug(  # type: ignore[empty-body]
        self: Diagram, other: Diagram, direction: V2_t
    ) -> Diagram: ...

    def beside_snug(  # type: ignore[empty-body]
        self: Diagram, other: Diagram, direction: V2_t
    ) -> Diagram: ...

    def juxtapose(  # type: ignore[empty-body]
        self: Diagram, other: Diagram, direction: V2_t
    ) -> Diagram: ...

    def atop(self: Diagram, other: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def above(self: Diagram, other: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def beside(  # type: ignore[empty-body]
        self: Diagram, other: Diagram, direction: V2_t
    ) -> Diagram: ...

    def frame(self, extra: tx.Floating) -> Diagram:  # type: ignore[empty-body]
        ...

    def pad(self, extra: tx.Floating) -> Diagram:  # type: ignore[empty-body]
        ...

    def scale_uniform_to_x(self, x: tx.Floating) -> Diagram:  # type: ignore[empty-body]
        ...

    def scale_uniform_to_y(self, y: tx.Floating) -> Diagram:  # type: ignore[empty-body]
        ...

    def align(self: Diagram, v: V2_t) -> Diagram:  # type: ignore[empty-body]
        ...

    def align_t(self: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def align_b(self: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def align_l(self: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def align_r(self: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def align_tl(self: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def align_tr(self: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def align_bl(self: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def align_br(self: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def snug(self: Diagram, v: V2_t) -> Diagram:  # type: ignore[empty-body]
        ...

    def center_xy(self: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def get_subdiagram(self, name: Name) -> Optional[Subdiagram]: ...

    def get_sub_map(  # type: ignore[empty-body]
        self, t: tx.Affine = Ident
    ) -> Dict[Name, List[Subdiagram]]: ...

    def with_names(  # type: ignore[empty-body]
        self,
        names: List[Name],
        f: Callable[[List[Subdiagram], Diagram], Diagram],
    ) -> Diagram: ...

    def _style(self, style: StyleHolder) -> Diagram:  # type: ignore[empty-body]
        ...

    def with_envelope(self, other: Diagram) -> Diagram:  # type: ignore[empty-body]
        ...

    def show_origin(self) -> Diagram:  # type: ignore[empty-body]
        ...

    def show_envelope(  # type: ignore[empty-body]
        self, phantom: bool = False, angle: int = 45
    ) -> Diagram: ...

    def compose(  # type: ignore[empty-body]
        self, envelope: Envelope, other: Optional[Diagram] = None
    ) -> Diagram: ...

    def to_list(  # type: ignore[empty-body]
        self, t: tx.Affine = Ident
    ) -> List[Diagram]: ...

    def accept(  # type: ignore[empty-body]
        self, visitor: DiagramVisitor[A, Any], args: Any
    ) -> A: ...
