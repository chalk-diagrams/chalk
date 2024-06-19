from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.transform import (
    Affine,
    P2_t,
    Ray,
    TraceDistances,
    Transformable,
    V2_t,
)

from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:

    from chalk.core import ApplyTransform, Primitive
    from chalk.types import Diagram
    from chalk.types import Traceable

class Trace(Monoid, Transformable):
    def __init__(self, f: Callable[[Ray], TraceDistances]) -> None:
        self.f = f

    def __call__(self, point: P2_t, direction: V2_t) -> TraceDistances:
        d, m = self.f(Ray(point, direction))
        ad = tx.X.np.argsort(d + (1 - m) * 1e10, axis=1)
        d = tx.X.np.take_along_axis(d, ad, axis=1)
        m = tx.X.np.take_along_axis(m, ad, axis=1)
        return d, m

    # Monoid
    @classmethod
    def empty(cls) -> Trace:
        return cls(lambda _: (tx.X.np.asarray([]), tx.X.np.asarray([])))

    def __add__(self, other: Trace) -> Trace:
        return Trace(lambda ray: tx.X.union(self.f(ray), other.f(ray)))

    @staticmethod
    def general_transform(t: Affine, fn) -> Trace:  # type: ignore
        t1 = tx.inv(t)

        def wrapped(
            ray: Ray,
        ) -> TraceDistances:
            trac, mask = fn(
                Ray(
                    t1 @ ray.pt[:, None, :, :],
                    t1 @ ray.v[:, None, :, :],
                )
            )
            return tx.X.union_axis((trac, mask), axis=-1)

        return Trace(wrapped)

    # Transformable
    def apply_transform(self, t: Affine) -> Trace:
        def apply(ray: Ray):  # type: ignore
            t, m = self.f(Ray(ray.pt[..., 0, :, :], ray.v[..., 0, :, :]))
            return t[..., None], m[..., None]

        return Trace.general_transform(t, apply)

    def trace_v(self, p: P2_t, v: V2_t) -> TraceDistances:
        v = tx.norm(v)
        dists, m = self(p, v)

        d = tx.X.np.sort(dists + (1 - m) * 1e10, axis=1)
        ad = tx.X.np.argsort(dists + (1 - m) * 1e10, axis=1)
        m = tx.X.np.take_along_axis(m, ad, axis=1)
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
        ps = tx.X.np.concatenate([ps, ps2], axis=1)
        m = tx.X.np.concatenate([m, m2], axis=1)
        ad = tx.X.np.argsort(ps + (1 - m) * 1e10, axis=1)
        ps = tx.X.np.take_along_axis(ps, ad, axis=1)
        m = tx.X.np.take_along_axis(m, ad, axis=1)
        return ps, m


class GetTrace(DiagramVisitor[Trace, Affine]):
    A_type = Trace

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Trace:
        new_transform = t @ diagram.transform

        if diagram.is_multi():
            # MultiPrimitive only work in jax mode.
            import jax

            def trace(ray: Ray) -> TraceDistances:
                def inner(shape : Traceable, transform: Affine) -> TraceDistances: 
                    trace = shape.get_trace().apply_transform(transform)
                    return trace(ray.pt, ray.v)

                r = jax.vmap(inner)(diagram.shape, diagram.transform)
                return tx.X.union_axis(r, axis=0)

            return Trace(trace)
        else:
            return diagram.shape.get_trace().apply_transform(new_transform)

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine
    ) -> Trace:
        return diagram.diagram.accept(self, t @ diagram.transform)


def get_trace(self: Diagram) -> Trace:
    return self.accept(GetTrace(), tx.X.ident)
