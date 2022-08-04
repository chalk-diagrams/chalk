from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional

from chalk.transform import (
    P2,
    V2,
    Affine,
    Transformable,
    apply_affine,
    remove_translation,
)
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import (
        ApplyName,
        ApplyStyle,
        ApplyTransform,
        Compose,
        Empty,
        Primitive,
    )
    from chalk.types import Diagram


SignedDistance = float
Ident = Affine.identity()


class Trace(Transformable):
    def __init__(self, f: Callable[[P2, V2], List[SignedDistance]]) -> None:
        self.f = f

    def __call__(self, point: P2, direction: V2) -> List[SignedDistance]:
        return self.f(point, direction)

    @classmethod
    def empty(cls) -> Trace:
        return cls(lambda point, direction: [])

    def __add__(self, other: Trace) -> Trace:
        return Trace(
            lambda point, direction: self(point, direction)
            + other(point, direction)
        )

    @staticmethod
    def mappend(trace1: Trace, trace2: Trace) -> Trace:
        return trace1 + trace2

    @staticmethod
    def concat(traces: Iterable[Trace]) -> Trace:
        return reduce(Trace.mappend, traces, Trace.empty())

    def apply_transform(self, t: Affine) -> Trace:  # type: ignore
        def wrapped(p: P2, d: V2) -> List[SignedDistance]:
            t1 = ~t
            return self(
                apply_affine(t1, p), apply_affine(remove_translation(t1), d)
            )

        return Trace(wrapped)

    def trace_v(self, p: P2, v: V2) -> Optional[V2]:
        v = v.scaled_to(1)
        dists = self(p, v)
        if dists:
            s, *_ = sorted(dists)
            return s * v
        else:
            return None

    def trace_p(self, p: P2, v: V2) -> Optional[P2]:
        u = self.trace_v(p, v)
        return p + u if u else None

    def max_trace_v(self, p: P2, v: V2) -> Optional[V2]:
        return self.trace_v(p, -v)

    def max_trace_p(self, p: P2, v: V2) -> Optional[P2]:
        u = self.max_trace_v(p, v)
        return p + u if u else None


class GetTrace(DiagramVisitor[Trace]):
    def visit_primitive(self, diagram: Primitive, t: Affine = Ident) -> Trace:
        new_transform = t * diagram.transform
        return diagram.shape.get_trace().apply_transform(new_transform)

    def visit_empty(self, diagram: Empty, t: Affine = Ident) -> Trace:
        return Trace.empty()

    def visit_compose(self, diagram: Compose, t: Affine = Ident) -> Trace:
        # TODO Should we cache the trace?
        return diagram.diagram1.accept(self, t=t) + diagram.diagram2.accept(
            self, t=t
        )

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine = Ident
    ) -> Trace:
        return diagram.diagram.accept(self, t=t * diagram.transform)

    def visit_apply_style(
        self, diagram: ApplyStyle, t: Affine = Ident
    ) -> Trace:
        return diagram.diagram.accept(self, t=t)

    def visit_apply_name(self, diagram: ApplyName, t: Affine = Ident) -> Trace:
        return diagram.diagram.accept(self, t=t)


def get_trace(self: Diagram, t: Affine = Ident) -> Trace:
    return self.accept(GetTrace(), t=t)
