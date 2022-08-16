"""
Contains arithmetic for arc calculations.
"""


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

from planar.py import Ray

import chalk.transform as tx
from chalk.envelope import Envelope
from chalk.shapes.segment import LocatedSegment, ray_circle_intersection
from chalk.trace import Trace
from chalk.transform import P2, V2, from_radians, unit_x, unit_y
from chalk.types import Enveloped, Traceable, TrailLike

if TYPE_CHECKING:
    from chalk.trail import Trail

Ident = tx.Affine.identity()
ORIGIN = P2(0, 0)

Degrees = float


def is_in_mod_360(x: Degrees, a: Degrees, b: Degrees) -> bool:
    """Checks if x ∈ [a, b] mod 360. See the following link for an
    explanation:
    https://fgiesen.wordpress.com/2015/09/24/intervals-in-modular-arithmetic/
    """
    return (x - a) % 360 <= (b - a) % 360


@dataclass
class LocatedArcSegment(Traceable, Enveloped, tx.Transformable):
    "A ellipse arc represented with the cetner parameterization"
    angle: float
    dangle: float

    # Ellipse is closed under affine.
    t: tx.Affine = tx.Affine.identity()

    # Parts to be careful about translation-invariance
    @property
    def p(self) -> P2:
        "Real start"
        return tx.apply_p2_affine(self.t, P2.polar(self.angle, 1))

    @property
    def q(self) -> P2:
        "Real end"
        return tx.apply_p2_affine(
            self.t, P2.polar(self.angle + self.dangle, 1)
        )

    @property
    def q_angle(self) -> float:
        return (self.q - self.center).angle

    @property
    def center(self) -> P2:
        "Real center"
        return tx.apply_p2_affine(self.t, P2(0, 0))

    @property
    def r_x(self) -> float:
        t2 = tx.remove_translation(self.t)
        return tx.apply_p2_affine(t2, unit_x).length

    @property
    def r_y(self) -> float:
        t2 = tx.remove_translation(self.t)
        return tx.apply_p2_affine(t2, unit_y).length

    @property
    def rot(self) -> float:
        t2 = tx.remove_translation(self.t)
        return tx.apply_p2_affine(t2, unit_x).angle

    def get_trace(self, t: tx.Affine = Ident) -> Trace:
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

    def get_envelope(self, t: tx.Affine = Ident) -> Envelope:
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
                return 1 / d.length
            else:
                # Case 2: P2 outside of arc
                x: float = max(d.dot(v1), d.dot(v2))
                return x

        return Envelope(wrapped).apply_transform(self.t)

    @staticmethod
    def arc_between(
        p: P2, q: P2, height: float
    ) -> Union[LocatedSegment, LocatedArcSegment]:

        h = abs(height)
        if h < 1e-3:
            return LocatedSegment(q - p, p)
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
        ret = (
            ArcSegment(flip * -from_radians(θ), flip * 2 * from_radians(θ))
            .scale(r)
            .rotate_rad(φ)
            .translate(d / 2, dy)
        )

        ret = ret.rotate(-diff.angle).translate_by(p)
        return ret


@dataclass
class ArcSegment(LocatedArcSegment, TrailLike):
    "A translation invariant version of Arc"

    def __post_init__(self) -> None:
        self.t = tx.Affine.translation(-self.p) * self.t
        assert self.p.x == 0, self.p
        assert self.p.y == 0, self.p

    def apply_transform(self, t: tx.Affine) -> ArcSegment:  # type: ignore
        t = tx.remove_translation(t)
        return ArcSegment(self.angle, self.dangle, t * self.t)

    def to_trail(self) -> Trail:
        from chalk.trail import Trail

        return Trail([self])

    @staticmethod
    def arc_between_trail(q: P2, height: float) -> Trail:
        segment = LocatedArcSegment.arc_between(P2(0, 0), q, height)
        if isinstance(segment, LocatedArcSegment):
            return ArcSegment(
                segment.angle, segment.dangle, segment.t
            ).to_trail()
        else:
            return segment.to_trail()


def arc_seg(q: V2, height: float) -> Trail:
    return ArcSegment.arc_between_trail(q, height)


def arc_seg_angle(angle: float, dangle: float) -> Trail:
    return ArcSegment(angle, dangle).to_trail()


# def arc_between(p:P2, q: P2, height: float) -> Path:
#     return ArcSegment.arc_between_trail(q - p, height).at(p)
