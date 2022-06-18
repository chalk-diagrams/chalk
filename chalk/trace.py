from functools import reduce
from typing import Callable, Iterable, List, Optional

from chalk.point import Point, Vector
from chalk.transform import Transform, Transformable, invert as invert_tx


SignedDistance = float


class Trace(Transformable):
    def __init__(
        self, f: Callable[[Point, Vector], List[SignedDistance]]
    ) -> None:
        self.f = f

    def __call__(
        self, point: Point, direction: Vector
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

    def apply_transform(self, t: Transform) -> "Trace":  # type: ignore
        def wrapped(p: Point, d: Vector) -> List[SignedDistance]:
            t1 = invert_tx(t)
            p1 = p.apply_transform(t1)
            d1 = d.apply_transform(t1)
            return self(p1, d1)

        return Trace(wrapped)

    def trace_v(self, p: Point, v: Vector) -> Optional[Vector]:
        dists = self(p, v)
        if dists:
            s, *_ = sorted(dists)
            return s * v
        else:
            return None

    def trace_p(self, p: Point, v: Vector) -> Optional[Point]:
        u = self.trace_v(p, v)
        return p + u if u else None


class Traceable:
    pass
