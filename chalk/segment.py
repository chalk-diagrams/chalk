from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from planar.py import Ray

import chalk.transform as tx
from chalk.envelope import Envelope
from chalk.trace import Trace
from chalk.transform import P2, V2

SignedDistance = float


@dataclass
class Segment(tx.Transformable):
    p: P2
    q: P2

    def get_trace(self) -> Trace:
        def f(point: P2, direction: V2) -> List[float]:
            ray = Ray(point, direction)
            inter = sorted(line_segment(ray, self))
            return inter

        return Trace(f)

    def get_envelope(self) -> Envelope:
        def f(d: V2) -> SignedDistance:
            x: float = max(d.dot(self.q), d.dot(self.p))
            return x

        return Envelope(f)

    def to_ray(self) -> "Ray":
        return Ray(self.p, self.q - self.p)

    @property
    def length(self) -> Any:
        return (self.q - self.p).length

    def apply_transform(self, t: tx.Affine) -> Segment:
        return Segment(tx.apply_affine(t, self.p), tx.apply_affine(t, self.q))

    def render_path(self, ctx) -> None:
        ctx.line_to(self.q.x, self.q.y)

    def render_svg_path(self) -> str:
        return f"L {self.q.x} {self.q.y}"

    def render_tikz_path(self, pts, pylatex) -> None:
        pts.append("--")
        pts.append(pylatex.TikZCoordinate(self.q.x, self.q.y))


def ray_ray_intersection(
    ray1: Ray, ray2: Ray
) -> Optional[Tuple[float, float]]:
    """Given two rays

    ray₁ = λ t . p₁ + t v₁
    ray₂ = λ t . p₂ + t v₂

    the function returns the parameters t₁ and t₂ at which the two rays meet,
    that is:

    ray₁ t₁ = ray₂ t₂

    """
    u = ray2.anchor - ray1.anchor
    x1 = ray1.direction.cross(ray2.direction)
    x2 = u.cross(ray1.direction)
    x3 = u.cross(ray2.direction)
    if x1 == 0 and x2 != 0:
        # parallel
        return None
    else:
        # intersecting or collinear
        return x3 / x1, x2 / x1


def line_segment(ray: Ray, segment: Segment) -> List[float]:
    """Given a ray and a segment, return the parameter `t` for which the ray
    meets the segment, that is:

    ray t₁ = segment.to_ray t₂, with t₂ ∈ [0, segment.length]

    Note: We need to consider the segment's length separately since `Ray`
    normalizes the direction to unit and hences looses this information. The
    length is important to determine whether the intersection point falls
    within the given segment.

    See also: https://github.com/danoneata/chalk/issues/91

    """
    ray_s = segment.to_ray()
    t = ray_ray_intersection(ray, ray_s)
    if not t:
        return []
    else:
        t1, t2 = t
        # the intersection point is given by any of the two expressions:
        # ray.anchor   + t1 * ray.direction
        # ray_s.anchor + t2 * ray_s.direction
        if 0 <= t2 <= segment.length:
            # intersection point is in segment
            return [t1]
        else:
            # intersection point outside
            return []


def ray_circle_intersection(ray: Ray, circle_radius: float) -> List[float]:
    """Given a ray and a circle centered at the origin, return the parameter t
    where the ray meets the circle, that is:

    ray t = circle θ

    The above equation is solved as follows:

    x + t v_x = r sin θ
    y + t v_y = r cos θ

    By squaring the equations and adding them we get

    (x + t v_x)² + (y + t v_y)² = r²,

    which is equivalent to the following equation:

    (v_x² + v_y²) t² + 2 (x v_x + y v_y) t + (x² + y² - r²) = 0

    This is a quadratic equation, whose solutions are well known.

    """
    p = ray.anchor

    a = ray.direction.length2
    b = 2 * (p.dot(ray.direction))
    c = p.length2 - circle_radius**2

    Δ = b**2 - 4 * a * c
    eps = 1e-6  # rounding error tolerance

    if Δ < -eps:
        # no intersection
        return []
    elif -eps <= Δ < eps:
        # tangent
        return [-b / (2 * a)]
    else:
        # the ray intersects at two points
        return [
            (-b - math.sqrt(Δ)) / (2 * a),
            (-b + math.sqrt(Δ)) / (2 * a),
        ]
