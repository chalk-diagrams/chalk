from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

from chalk.monoid import Monoid
from chalk.transform import (
    P2_t,
    V2_t,
    Affine,
    Transformable,
    Scalar
)
import chalk.transform as tr
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Primitive
    from chalk.types import Diagram


SignedDistance = Scalar
Ident = tr.ident


class Trace(Monoid, Transformable):
    def __init__(self, f: Callable[[P2_t, V2_t], List[SignedDistance]]) -> None:
        self.f = f

    def __call__(self, point: P2_t, direction: V2_t) -> List[SignedDistance]:
        return self.f(point, direction)

    # Monoid
    @classmethod
    def empty(cls) -> Trace:
        return cls(lambda point, direction: [])

    def __add__(self, other: Trace) -> Trace:
        return Trace(
            lambda point, direction: self(point, direction)
            + other(point, direction)
        )

    # Transformable
    def apply_transform(self, t: Affine) -> Trace:
        def wrapped(p: P2_t, d: V2_t) -> List[SignedDistance]:
            t1 = tr.inv(t)
            return self(
                t1 @ p, tr.remove_translation(t1) @ d
            )

        return Trace(wrapped)

    def trace_v(self, p: P2_t, v: V2_t) -> Optional[V2_t]:
        v = tr.norm(v)
        dists = self(p, v)
        if dists:
            s, *_ = sorted(dists)
            return s * v
        else:
            return None

    def trace_p(self, p: P2_t, v: V2_t) -> Optional[P2_t]:
        u = self.trace_v(p, v)
        return p + u if u else None

    def max_trace_v(self, p: P2_t, v: V2_t) -> Optional[V2_t]:
        return self.trace_v(p, -v)

    def max_trace_p(self, p: P2_t, v: V2_t) -> Optional[P2_t]:
        u = self.max_trace_v(p, v)
        return p + u if u else None


class GetTrace(DiagramVisitor[Trace, Affine]):
    A_type = Trace

    def visit_primitive(self, diagram: Primitive, t: Affine = Ident) -> Trace:
        new_transform = t @ diagram.transform
        return diagram.shape.get_trace().apply_transform(new_transform)

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine = Ident
    ) -> Trace:
        return diagram.diagram.accept(self, t @ diagram.transform)


def get_trace(self: Diagram, t: Affine = Ident) -> Trace:
    return self.accept(GetTrace(), t)
