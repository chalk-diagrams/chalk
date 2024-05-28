"""
Contains arithmetic for arc calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Union


import chalk.transform as tx
from chalk.transform import Affine
from chalk.envelope import Envelope
from chalk.trace import Trace
from chalk.transform import P2_t, V2_t, Scalar
from chalk.types import Enveloped, Traceable, TrailLike

if TYPE_CHECKING:
    from chalk.trail import Trail
    from jaxtyping import Float, Bool, Array

Degrees = tx.Scalars

@dataclass
class Segment(TrailLike):
    "A ellipse arc with starting angle and the delta."
    transform: Float[Array, "#B 3 3"]
    angles: Float[Array, "#B 2"]

    def to_trail(self) -> Trail:
        from chalk.trail import Trail
 
        return Trail(self)
    
    # Transformable
    def apply_transform(self, t: Affine) -> Segment:
        return Segment(t @ self.transform, self.angles)
    
    def __add__(self, other: Segment) -> Segment:
        return Segment(tx.np.concat([self.transform, other.transform], axis=0), 
                       tx.np.concat([self.angles, other.angles], axis=0))
                     
    @property
    def t(self) -> Float[Array, "#B 3 3"]:
        return self.transform

    @property
    def q(self) -> Float[Array, "#B 3 1"]:
        print(self.angles.sum(-1), self.angles.shape)
        return self.t @ tx.to_point(tx.polar(self.angles.sum(-1)))
    
    @property
    def angle(self) -> Float[Array, "#B"]:
        return self.angles[:, 0]

    @property
    def dangle(self) -> Float[Array, "#B"]:
        return self.angles[:, 1]

    @property
    def r_x(self) -> Float[Array, "#B"]:
        return tx.length(self.t @ tx.unit_x)

    @property
    def r_y(self) -> Float[Array, "#B"]:
        return tx.length(self.t @ tx.unit_y)

    @property
    def rot(self) -> Float[Array, "#B"]:
        return tx.angle(self.t @ tx.unit_x)

def seg(offset: V2_t) -> Trail:
    return arc_seg(offset, 1e-4)

def is_in_mod_360(x: Degrees, a: Degrees, b: Degrees) -> tx.Mask:
    """Checks if x ∈ [a, b] mod 360. See the following link for an
    explanation:
    https://fgiesen.wordpress.com/2015/09/24/intervals-in-modular-arithmetic/
    """
    return (x - a) % 360 <= (b - a) % 360


def arc_between(
    p: P2_t, q: P2_t, height: tx.Scalars
) -> Segment:

    h = abs(height)
    # if h < 1e-3:
    #     return LocatedSegment(q - p, p)
    d = tx.length(q - p)
    # Determine the arc's angle θ and its radius r
    θ = tx.np.acos((d**2 - 4.0 * h**2) / (d**2 + 4.0 * h**2))
    r = d / (2 * tx.np.sin(θ))

    if height > 0:
        # bend left
        φ = +tx.np.pi / 2
        dy = r - h
        flip = 1
    else:
        # bend right
        φ = -tx.np.pi / 2
        dy = h - r
        flip = -1

    diff = q - p
    angles = tx.np.array([flip * -tx.from_radians(θ), 
                          flip * 2 * tx.from_radians(θ)], float).reshape(1, 2)
    ret = tx.translation(p) @ tx.rotation(-tx.rad(diff)) @  tx.translation(tx.V2(d / 2, dy)) @ tx.rotation(φ) @ tx.scale(tx.V2(r, r))
    return Segment(ret, angles)

def arc_envelope(angle_offset: Float[Array, "#B 2"]) -> Callable[[V2_t], tx.Scalars]:
    "Trace is done as simple arc and transformed"
    angle0_deg = angle_offset[..., 0]
    angle1_deg = angle0_deg + angle_offset[..., 1]

    v1 = tx.polar(angle0_deg)
    v2 = tx.polar(angle1_deg)
    def wrapped(d: V2_t) -> tx.Scalars:
        is_circle = abs(angle0_deg - angle1_deg) >= 360
        q =  tx.np.where(
            is_circle | is_in_mod_360(
            tx.angle(d),
            tx.np.minimum(angle0_deg, angle1_deg),
            tx.np.maximum(angle0_deg, angle1_deg),
            ), 
            # Case 1: P2 at arc
            1 / tx.length(d), 
            # Case 2: P2 outside of arc
            tx.np.maximum(tx.dot(d, v1), tx.dot(d, v2))
        )
        return q
    return wrapped

def arc_seg(q: V2_t, height: tx.Floating) -> Trail:
    return arc_between_trail(q, tx.ftos(height))

def arc_seg_angle(angle: tx.Floating, dangle: tx.Floating) -> Trail:
    arc_p = tx.to_point(tx.polar(angle))
    q = Segment(tx.translation(-arc_p), 
                tx.np.array([angle, dangle], float)[None])
    return q.to_trail()


def arc_between_trail(q: P2_t, height: tx.Scalars) -> Trail:
    from chalk.trail import Trail

    return arc_between(tx.P2(0, 0), q, height).to_trail()

# @dataclass
# class LocatedArcSegment(Traceable, Enveloped, tx.Transformable):
#     "A ellipse arc with starting angle and the delta."
#     angle: tx.Scalar
#     dangle: tx.Scalar

#     # Ellipse is closed under affine.
#     t: tx.Affine = field(default_factory=lambda :tx.ident)

#     # Parts to be careful about translation-invariance
#     @property
#     def p(self) -> P2_t:
#         "Start point"
#         return self.t @ tx.to_point(tx.polar(self.angle))

#     @property
#     def q(self) -> P2_t:
#         "End point"
#         return self.t @ tx.to_point(tx.polar(self.angle + self.dangle))

#     @property
#     def q_angle(self) -> Scalar:
#         return tx.angle(self.q - self.center)

#     @property
#     def center(self) -> P2_t:
#         "Center of the implied ellipse"
#         return self.t @ tx.P2(0, 0)

#     @property
#     def r_x(self) -> Scalar:
#         return tx.length(self.t @ tx.unit_x)

#     @property
#     def r_y(self) -> Scalar:
#         return tx.length(self.t @ tx.unit_y)

#     @property
#     def rot(self) -> Scalar:
#         return tx.angle(self.t @ tx.unit_x)

#     def get_trace(self, t: tx.Affine = tx.ident) -> Trace:
#         "Trace is done as simple arc and transformed"
#         angle0_deg = self.angle
#         angle1_deg = self.angle + self.dangle

#         def f(p: P2_t, v: V2_t) -> List[Scalar]:
#             return sorted(
#                 [
#                     d / tx.length(v)
#                     for d in tx.ray_circle_intersection(p, v, tx.ftos(1))
#                     if is_in_mod_360(
#                         tx.angle((d * v) + tx.polar(self.angle)),
#                         tx.np.minimum(angle0_deg, angle1_deg),
#                         tx.np.maximum(angle0_deg, angle1_deg),
#                     )
#                 ]
#             )

#         return Trace(f).apply_transform(self.t)

#     def get_envelope(self, t: tx.Affine = tx.ident) -> Envelope:
#         "Trace is done as simple arc and transformed"
#         angle0_deg = self.angle
#         angle1_deg = self.angle + self.dangle

#         v1 = tx.polar(angle0_deg)
#         v2 = tx.polar(angle1_deg)

#         def wrapped(d: V2_t) -> Scalar:
#             is_circle = abs(angle0_deg - angle1_deg) >= 360
#             return tx.np.where(
#                 is_circle | is_in_mod_360(
#                 tx.angle(d),
#                 tx.np.minimum(angle0_deg, angle1_deg),
#                 tx.np.maximum(angle0_deg, angle1_deg),
#                ), 
#                 # Case 1: P2 at arc
#                 1 / tx.length(d), 
#                 # Case 2: P2 outside of arc
#                 tx.np.maximum(d @ v1, d @ v2)
#             )
#         return Envelope(wrapped).apply_transform(self.t)

  


# @dataclass
# class ArcSegment(LocatedArcSegment, TrailLike):
#     "A translation invariant version of Arc"

#     def __post_init__(self) -> None:
#         self.t = tx.translation(-self.p) @ self.t
#         assert self.p[0] == 0, self.p
#         assert self.p[1] == 0, self.p

#     def apply_transform(self, t: tx.Affine) -> ArcSegment:
#         t = tx.remove_translation(t)
#         return ArcSegment(self.angle, self.dangle, t @ self.t)

#     def to_trail(self) -> Trail:
#         from chalk.trail import Trail

#         return Trail([self])

#     @staticmethod
#     def arc_between_trail(q: P2_t, height: Scalar) -> Trail:
#         segment = LocatedArcSegment.arc_between(tx.P2(0, 0), q, height)
#         if isinstance(segment, LocatedArcSegment):
#             return ArcSegment(
#                 segment.angle, segment.dangle, segment.t
#             ).to_trail()
#         else:
#             return segment.to_trail()

#     def reverse(self) -> ArcSegment:
#         angle = self.angle + self.dangle
#         dangle = -self.dangle
#         return ArcSegment(angle, dangle).apply_transform(self.t)






# # def arc_between(p:P2, q: P2, height: float) -> Path:
# #     return ArcSegment.arc_between_trail(q - p, height).at(p)
