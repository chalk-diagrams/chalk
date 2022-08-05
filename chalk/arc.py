"""
Contains arithmetic for arc calculations.
"""


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, NewType

from planar.py import Ray

import chalk.transform as tx
from chalk.envelope import Envelope
from chalk.segment import ray_circle_intersection
from chalk.trace import Trace
from chalk.transform import P2, V2, from_radians, to_radians, unit_x, unit_y

Degrees = float


def is_in_mod_360(x: Degrees, a: Degrees, b: Degrees) -> bool:
    """Checks if x ∈ [a, b] mod 360. See the following link for an
    explanation:
    https://fgiesen.wordpress.com/2015/09/24/intervals-in-modular-arithmetic/
    """
    return (x - a) % 360 <= (b - a) % 360


@dataclass
class ArcSegment(tx.Transformable):
    "A ellipse arc represented with the cetner parameterization"
    angle: float
    dangle: float

    # Ellipse is closed under affine.
    t: tx.Transform = tx.Affine.identity()

    # Lazy center parameterization.
    @property
    def p(self) -> P2:
        "Real start"
        return tx.apply_affine(self.t, P2.polar(self.angle, 1))

    @property
    def q(self) -> P2:
        "Real end"
        return tx.apply_affine(self.t, P2.polar(self.angle + self.dangle, 1))

    @property
    def r_x(self) -> float:
        t2 = tx.remove_translation(self.t)
        return tx.apply_affine(t2, unit_x).length

    @property
    def r_y(self) -> float:
        t2 = tx.remove_translation(self.t)
        return tx.apply_affine(t2, unit_y).length

    @property
    def rot(self) -> float:
        t2 = tx.remove_translation(self.t)
        return tx.apply_affine(t2, unit_x).angle

    @property
    def center(self) -> P2:
        "Real center"
        return tx.apply_affine(self.t, P2(0, 0))

    def apply_transform(self, t: tx.Affine) -> ArcSegment:
        return ArcSegment(self.angle, self.dangle, t * self.t)

    def get_trace(self) -> Trace:
        "Trace is done as simple arc and transformed"
        angle0_deg = self.angle
        angle1_deg = self.angle + self.dangle

        def f(p: P2, v: V2) -> List[float]:
            ray = Ray(p, v)
            return sorted(
                [
                    d / v.length
                    for d in ray_circle_intersection(ray, 1)
                    if is_in_mod_360(
                        ((d * v) + P2.polar(self.angle, 1)).angle,
                        min(angle0_deg, angle1_deg),
                        max(angle0_deg, angle1_deg),
                    )
                ]
            )

        return Trace(f).apply_transform(self.t)

    def get_envelope(self) -> Envelope:
        "Trace is done as simple arc and transformed"
        angle0_deg = self.angle
        angle1_deg = self.angle + self.dangle

        v1 = V2.polar(angle0_deg, 1)
        v2 = V2.polar(angle1_deg, 1)

        def wrapped(d: V2) -> float:
            is_circle = abs(angle0_deg - angle1_deg) >= 360
            if is_circle or is_in_mod_360(
                d.angle,
                min(angle0_deg, angle1_deg),
                max(angle0_deg, angle1_deg),
            ):
                # Case 1: P2 at arc
                return 1 / d.length  # type: ignore
            else:
                # Case 2: P2 outside of arc
                x: float = max(d.dot(v1), d.dot(v2))
                return x

        return Envelope(wrapped).apply_transform(self.t)

    @staticmethod
    def arc_between(p: P2, q: P2, height: float) -> ArcSegment:
        h = abs(height)
        d = (q - p).length
        # Determine the arc's angle θ and its radius r
        θ = math.acos((d**2 - 4.0 * h**2) / (d**2 + 4.0 * h**2))
        r = d / (2 * math.sin(θ))

        if height > 0:
            # bend left
            φ = +math.pi / 2
            dy = r - h
            flip = 1
        else:
            # bend right
            φ = -math.pi / 2
            dy = h - r
            flip = -1

        diff = q - p
        r = (
            ArcSegment(flip * -from_radians(θ), flip * 2 * from_radians(θ))
            .scale(r)
            .rotate_rad(φ)
            .translate(d / 2, dy)
        )

        r = r.rotate(-diff.angle).translate_by(p)
        return r

    def render_path(self, ctx) -> None:
        end = self.angle + self.dangle

        ctx.new_sub_path()
        ctx.save()
        matrix = tx.to_cairo(self.t)
        ctx.transform(matrix)
        if self.dangle < 0:
            ctx.arc_negative(
                0.0, 0.0, 1.0, to_radians(self.angle), to_radians(end)
            )
        else:
            ctx.arc(0.0, 0.0, 1.0, to_radians(self.angle), to_radians(end))
        ctx.restore()

    def render_svg_path(self) -> str:
        "https://www.w3.org/TR/SVG/implnote.html#ArcConversionCenterToEndpoint"
        f_A = 1 if abs(self.dangle) > 180 else 0
        f_S = 1 if self.dangle > 0 else 0
        return f"A {self.r_x} {self.r_y} {self.rot} {f_A} {f_S} {self.q.x} {self.q.y}"

    def reflect_y(self) -> None:
        assert False

    def reflect_x(self) -> None:
        assert False

    def render_tikz_path(self, pts, pylatex) -> None:
        start = (self.p - self.center).angle
        end = (self.q - self.center).angle
        if self.dangle < 0 and end > start:
            end = end - 360
        if self.dangle > 0 and end < start:
            end = end + 360
        end_ang = end - self.rot
        pts._arg_list.append(
            pylatex.TikZUserPath(
                f"{{[rotate={self.rot}] arc[start angle={start-self.rot}, end angle={end_ang}, x radius={self.r_x}, y radius ={self.r_y}]}}"
            )
        )
