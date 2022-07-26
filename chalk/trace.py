from __future__ import annotations

from functools import reduce
from typing import Callable, Iterable, List, Optional

from chalk.transform import (
    P2,
    V2,
    Affine,
    Transformable,
    apply_affine,
    remove_translation,
)

SignedDistance = float


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
        dists = [d for d in dists if d >= 0.0] 
        if dists:
            s, *_ = sorted(dists)
            return s * v
        else:
            return None

    def trace_p(self, p: P2, v: V2) -> Optional[P2]:
        v = v.scaled_to(1)
        u = self.trace_v(p, v)
        return p + u if u else None


class Traceable:
    pass
