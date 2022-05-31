from functools import reduce
from typing import Callable, Iterable, List

from chalk.point import Point, Vector
from chalk.transform import Transform, Transformable


SignedDistance = float


class Trace(Transformable):
    def __init__(self, f: Callable[[Point, Vector], List[SignedDistance]]) -> None:
        self.f = f

    def __call__(self, point: Point, direction: Vector) -> List[SignedDistance]:
        return self.f(point, direction)

    @classmethod
    def empty(cls) -> "Trace":
        return cls(lambda point, direction: [])

    def __add__(self, other: "Trace") -> "Trace":
        return Trace(
            lambda point, direction: self(point, direction) + other(point, direction)
        )

    @staticmethod
    def mappend(trace1: "Trace", trace2: "Trace") -> "Trace":
        return trace1 + trace2

    @staticmethod
    def concat(traces: Iterable["Trace"]) -> "Trace":
        return reduce(Trace.mappend, traces, Trace.empty())

    def apply_transform(self, t: Transform) -> "Trace":  # type: ignore
        def wrapped(p: Point, d: Vector) -> List[SignedDistance]:
            import pdb; pdb.set_trace()

        return Trace(wrapped)


class Traceable:
    pass
