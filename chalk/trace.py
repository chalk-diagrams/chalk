from functools import reduce
from typing import Callable, Iterable, List, Optional

from planar import Point, Vec2
from chalk.transform import Affine, Transformable, apply_affine, remove_translation

SignedDistance = float


class Trace(Transformable):
    def __init__(
        self, f: Callable[[Point, Vec2], List[SignedDistance]]
    ) -> None:
        self.f = f

    def __call__(
        self, point: Point, direction: Vec2
    ) -> List[SignedDistance]:
        return self.f(point, direction)

    @classmethod
    def empty(cls) -> "Trace":
        return cls(lambda point, direction: [])

    def __add__(self, other: "Trace") -> "Trace":
        return Trace(
            lambda point, direction: self(point, direction)
            + other(point, direction)
        )

    @staticmethod
    def mappend(trace1: "Trace", trace2: "Trace") -> "Trace":
        return trace1 + trace2

    @staticmethod
    def concat(traces: Iterable["Trace"]) -> "Trace":
        return reduce(Trace.mappend, traces, Trace.empty())

    def apply_transform(self, t: Affine) -> "Trace":  # type: ignore
        def wrapped(p: Point, d: Vec2) -> List[SignedDistance]:
            t1 = ~t
            return self(apply_affine(t1, p), apply_affine(remove_translation(t1), d))

        return Trace(wrapped)

    def trace_v(self, p: Point, v: Vec2) -> Optional[Vec2]:
        v = v.scaled_to(1)
        dists = self(p, v)
        if dists:
            s, *_ = sorted(dists)
            return s * v
        else:
            return None

    def trace_p(self, p: Point, v: Vec2) -> Optional[Point]:
        v = v.scaled_to(1)
        u = self.trace_v(p, v)
        return p + u if u else None


class Traceable:
    pass
