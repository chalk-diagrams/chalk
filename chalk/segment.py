import math

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
            inter = sorted(line_segment(line, self))
            return inter

        return Trace(f)

    def to_line(self) -> "Line":
        return Line(self.p, self.q - self.p)


@dataclass
class Line:
    p: Point
    v: Vector


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


def line_segment(line: Line, segment: Segment) -> List[float]:
    """Given a line and a segment, return the parameter t for which the line
    meets the segment, that is:

    line t = line t', with t' ∈ [0, 1]

    """
    line_s = segment.to_line()
    t = line_line_intersection(line, line_s)
    if not t:
        return []
    else:
        t1, t2 = t
        if 0 <= t2 <= 1:
            # p = line_s.p + t2 * line_s.v
            return [t1]
        else:
            return []


def line_circle_intersection(line: Line, circle_radius: float) -> List[float]:
    """Given a line and a circle centered at the origin, return the parameter t
    where the line meets the circle, that is:

    line t = circle θ

    The above equation is solved as follows:

    x + t v_x = r sin θ
    y + t v_y = r cos θ

    By squaring the equations and adding them we get

    (x + t v_x)² + (y + t v_y)² = r²,

    which is equivalent to the following equation:

    (v_x² + v_y²) t² + 2 (x v_x + y v_y) t + (x² + y² - r²) = 0

    This is a quadratic equation, whose solutions are well known.

    """
    a = line.v.dx**2 + line.v.dy**2
    b = 2 * (line.p.x * line.v.dx + line.p.y * line.v.dy)
    c = line.p.x**2 + line.p.y**2 - circle_radius**2

    Δ = b**2 - 4 * a * c
    eps = 1e-6  # rounding error tolerance

    if Δ < -eps:
        # no intersection
        return []
    elif -eps <= Δ < eps:
        # tagent
        return [-b / (2 * a)]
    else:
        # the line intersects at two points
        return [
            (-b - math.sqrt(Δ)) / (2 * a),
            (-b + math.sqrt(Δ)) / (2 * a),
        ]
