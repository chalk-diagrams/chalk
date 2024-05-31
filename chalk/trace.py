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
from dataclasses import dataclass
import chalk.transform as tx
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Primitive
    from chalk.types import Diagram
    from jaxtyping import Bool, Array
    TraceDistances = Tuple[Float[Array, f"#B {SIZE}"], 
                           Bool[Array, f"#B {SIZE}"]]



@dataclass
class Ray:
    pt : P2_t
    v : V2_t

SIZE = 64


class Trace(Monoid, Transformable):
    FILL = -999

    def __init__(self, f: Callable[[Ray], TraceDistances]) -> None:
        self.f = f

    def __call__(self, point: P2_t, direction: V2_t) -> TraceDistances:
        return self.f(Ray(point, direction))



    # Monoid
    @classmethod
    def empty(cls) -> Trace:
        return cls(lambda _: (tx.np.array(), tx.np.array()))

    def __add__(self, other: Trace) -> Trace:
        return Trace(
            lambda ray: tx.union(self.f(ray), other.f(ray))
        )

    # Transformable
    def apply_transform(self, t: Affine) -> Trace:
        t1 = tx.inv(t)
        def wrapped(ray: Ray) -> TraceDistances:
            return self.f(Ray(
                t1 @ ray.pt, 
                tx.remove_translation(t1) @ ray.v)
            )

        return Trace(wrapped)

    def trace_v(self, p: P2_t, v: V2_t) -> Optional[V2_t]:
        v = tx.norm(v)
        dists, m = self(p, v)
        print("dists", dists, m)
        if dists.shape[1] > 0:
            d = tx.np.sort(dists, axis=1)
            s, *_ = d[:, 0]
            return s * v
        else:
            return None
        
    def trace_p(self, p: P2_t, v: V2_t) -> Optional[P2_t]:
        u = self.trace_v(p, v)
        return p + u if u is not None else None

    def max_trace_v(self, p: P2_t, v: V2_t) -> Optional[V2_t]:
        return self.trace_v(p, -v)

    def max_trace_p(self, p: P2_t, v: V2_t) -> Optional[P2_t]:
        u = self.max_trace_v(p, v)
        return p + u if u is not None else None


class GetTrace(DiagramVisitor[Trace, Affine]):
    A_type = Trace

    def visit_primitive(self, diagram: Primitive, t: Affine = tx.ident) -> Trace:
        new_transform = t @ diagram.transform
        return diagram.shape.get_trace().apply_transform(new_transform)

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine = tx.ident
    ) -> Trace:
        return diagram.diagram.accept(self, t @ diagram.transform)


def get_trace(self: Diagram, t: Affine = tx.ident) -> Trace:
    return self.accept(GetTrace(), t)
