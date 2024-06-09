from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.transform import Affine, P2_t, Ray, Transformable, V2_t
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from jaxtyping import Array, Bool, Float

    from chalk.core import ApplyTransform, Primitive
    from chalk.types import Diagram

    SIZE = 64
    TraceDistances = Tuple[
        Float[Array, f"#B {SIZE}"], Bool[Array, f"#B {SIZE}"]
    ]


class Trace(Monoid, Transformable):
    def __init__(self, f: Callable[[Ray], TraceDistances]) -> None:
        self.f = f

    def __call__(self, point: P2_t, direction: V2_t) -> TraceDistances:
        d, m = self.f(Ray(point, direction))
        ad = tx.np.argsort(d + (1 - m) * 1e10, axis=1)
        d = tx.np.take_along_axis(d, ad, axis=1)
        m = tx.np.take_along_axis(m, ad, axis=1)
        return d, m

    # Monoid
    @classmethod
    def empty(cls) -> Trace:
        return cls(lambda _: (tx.np.array([]), tx.np.array([])))

    def __add__(self, other: Trace) -> Trace:
        return Trace(lambda ray: tx.union(self.f(ray), other.f(ray)))

    # Transformable
    def apply_transform(self, t: Affine) -> Trace:
        t1 = tx.inv(t)

        def wrapped(ray: Ray) -> TraceDistances:
            return self.f(Ray(t1 @ ray.pt, tx.remove_translation(t1) @ ray.v))

        return Trace(wrapped)

    def trace_v(self, p: P2_t, v: V2_t) -> TraceDistances:
        v = tx.norm(v)
        dists, m = self(p, v)

        d = tx.np.sort(dists + (1 - m) * 1e10, axis=1)
        ad = tx.np.argsort(dists + (1 - m) * 1e10, axis=1)
        m = tx.np.take_along_axis(m, ad, axis=1)
        s = d[:, 0:1]
        return s[..., None] * v, m[:, 0]

    def trace_p(self, p: P2_t, v: V2_t) -> TraceDistances:
        u, m = self.trace_v(p, v)
        return p + u, m

    def max_trace_v(self, p: P2_t, v: V2_t) -> TraceDistances:
        return self.trace_v(p, -v)

    def max_trace_p(self, p: P2_t, v: V2_t) -> TraceDistances:
        u, m = self.max_trace_v(p, v)
        return p + u, m

    @staticmethod
    def combine(p1: TraceDistances, p2: TraceDistances) -> TraceDistances:
        ps, m = p1
        ps2, m2 = p2
        ps = tx.np.concatenate([ps, ps2], axis=1)
        m = tx.np.concatenate([m, m2], axis=1)
        ad = tx.np.argsort(ps + (1 - m) * 1e10, axis=1)
        ps = tx.np.take_along_axis(ps, ad, axis=1)
        m = tx.np.take_along_axis(m, ad, axis=1)
        return ps, m


class GetTrace(DiagramVisitor[Trace, Affine]):
    A_type = Trace

    def visit_primitive(
        self, diagram: Primitive, t: Affine = tx.ident
    ) -> Trace:
        new_transform = t @ diagram.transform
        return diagram.shape.get_trace().apply_transform(new_transform)

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine = tx.ident
    ) -> Trace:
        return diagram.diagram.accept(self, t @ diagram.transform)


def get_trace(self: Diagram, t: Affine = tx.ident) -> Trace:
    return self.accept(GetTrace(), t)
