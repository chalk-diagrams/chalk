from dataclasses import dataclass
from typing import List, Optional, Tuple

from chalk.point import Point, Vector
from chalk.trace import Trace


@dataclass
class Segment:
    p: Point
    q: Point

    def get_trace(self) -> Trace:
        def f(point: Point, direction: Vector) -> List[float]:
            line = Line(point, direction)
            inter = sorted(d for d, _, _ in line_segment(line, self))
            return inter

        return Trace(f)

    def to_line(self) -> "Line":
        return Line(self.p, self.q - self.p)


@dataclass
class Line:
    p: Point
    v: Vector


def line_segment(
    line: Line, segment: Segment
) -> List[Tuple[float, float, Point]]:
    line_s = segment.to_line()
    t = line_line_intersection(line, line_s)
    if not t:
        return []
    else:
        t1, t2 = t
        if 0 <= t2 <= 1:
            p = line_s.p + t2 * line_s.v
            return [(t1, t2, p)]
        else:
            return []


def line_line_intersection(
    line1: Line, line2: Line
) -> Optional[Tuple[float, float]]:
    """Given two lines

    line₁ = λ t . p₁ + t v₁
    line₂ = λ t . p₂ + t v₂

    the function returns the parameters t₁ and t₂ at which the two lines meet,
    that is:

    line₁ t₁ = line₂ t₂

    """
    u = line2.p - line1.p
    x1 = line1.v.cross(line2.v)
    x2 = u.cross(line1.v)
    x3 = u.cross(line2.v)
    if x1 == 0 and x2 != 0:
        # parallel
        return None
    else:
        # intersecting or collinear
        return x3 / x1, x2 / x1
