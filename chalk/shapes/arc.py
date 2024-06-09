"""
Contains arithmetic for arc calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Tuple

import chalk.transform as tx
from chalk.transform import Affine, Angles, P2_t, Scalars, V2_t
from chalk.types import TrailLike

if TYPE_CHECKING:

    from chalk.trail import Trail

Degrees = tx.Scalars


@dataclass
class Segment(TrailLike):
    "A batch of ellipse arcs with starting angle and the delta."
    transform: Affine
    angles: Angles

    def __init__(self, transform: Affine, angles: Angles):
        self.transform = transform.reshape(-1, 3, 3)
        self.angles = angles.reshape(-1, 2)

    def split(self, i: int) -> Segment:
        return Segment(self.transform[i], self.angles[i])

    def to_trail(self) -> Trail:
        from chalk.trail import Trail

        return Trail(self)

    def get(self, i: int) -> Segment:
        return Segment(transform=self.transform[i], angles=self.angles[i])

    # Transformable
    def apply_transform(self, t: Affine) -> Segment:
        return Segment(t @ self.transform, self.angles)

    def __add__(self, other: Segment) -> Segment:
        trans = [self.transform, other.transform]
        angles = [self.angles, other.angles]
        return Segment(
            tx.np.concatenate(trans, axis=-3),
            tx.np.concatenate(angles, axis=-2),
        )

    @property
    def t(self) -> Affine:
        return self.transform

    @property
    def q(self) -> P2_t:
        return self.t @ tx.to_point(tx.polar(self.angles.sum(-1)))

    @property
    def angle(self) -> Scalars:
        return self.angles[..., 0]

    @property
    def dangle(self) -> Scalars:
        return self.angles[..., 1]

    @property
    def r_x(self) -> Scalars:
        return tx.length(self.t @ tx.unit_x)

    @property
    def r_y(self) -> Scalars:
        return tx.length(self.t @ tx.unit_y)

    @property
    def rot(self) -> Scalars:
        return tx.angle(self.t @ tx.unit_x)

    @property
    def center(self) -> P2_t:
        return self.t @ tx.P2(0, 0)


def seg(offset: V2_t) -> Trail:
    return arc_seg(offset, 1e-3)


def is_in_mod_360(x: Degrees, a: Degrees, b: Degrees) -> tx.Mask:
    """Checks if x ∈ [a, b] mod 360. See the following link for an
    explanation:
    https://fgiesen.wordpress.com/2015/09/24/intervals-in-modular-arithmetic/
    """
    return (x - a) % 360 <= (b - a) % 360


def arc_between(p: P2_t, q: P2_t, height: tx.Scalars) -> Segment:

    h = abs(height)
    # if h < 1e-3:
    #     return LocatedSegment(q - p, p)
    d = tx.length(q - p)
    # Determine the arc's angle θ and its radius r
    θ = tx.np.arccos((d**2 - 4.0 * h**2) / (d**2 + 4.0 * h**2))
    r = d / (2 * tx.np.sin(θ))

    # bend left
    bl = height > 0
    φ = tx.np.where(bl, +tx.np.pi / 2, -tx.np.pi / 2)
    dy = tx.np.where(bl, r - h, h - r)
    flip = tx.np.where(bl, 1, -1)

    diff = q - p
    angles = tx.np.array(
        [flip * -tx.from_radians(θ), flip * 2 * tx.from_radians(θ)], float
    ).reshape(1, 2)
    ret = (
        tx.translation(p)
        @ tx.rotation(-tx.rad(diff))
        @ tx.translation(tx.V2(d / 2, dy))
        @ tx.rotation(φ)
        @ tx.scale(tx.V2(r, r))
    )
    return Segment(ret, angles)


def arc_envelope(
    angle_offset: Float[Array, "#B 2"]
) -> Callable[[Float[Array, "#A 1 3 1"]], Float[Array, "#A #B 3 1"]]:
    "Trace is done as simple arc and transformed"
    angle0_deg = angle_offset[..., 0]
    angle1_deg = angle0_deg + angle_offset[..., 1]

    is_circle = abs(angle0_deg - angle1_deg) >= 360
    low = tx.np.minimum(angle0_deg, angle1_deg)
    high = tx.np.maximum(angle0_deg, angle1_deg)
    check = (low - high) % 360

    v1 = tx.polar(angle0_deg)
    v2 = tx.polar(angle1_deg)

    def wrapped(d: Float[Array, "#A 1 3 1"]) -> Float[Array, "#A #B 3 1"]:
        return tx.np.where(
            (is_circle | (((tx.angle(d) - high) % 360) > check)),
            # Case 1: P2 at arc
            1 / tx.length(d),
            # Case 2: P2 outside of arc
            tx.np.maximum(tx.dot(d, v1), tx.dot(d, v2)),
        )

    return wrapped


OFFSET = 0.0


def set_offset(v: float) -> None:
    global OFFSET
    OFFSET = v


def arc_trace(
    angle_offset: Float[Array, "#B 2"]
) -> Callable[
    [tx.Ray], Tuple[Float[Array, "#A #B 2"], Bool[Array, "#A #B 2"]]
]:
    "Trace is done as simple arc and transformed"
    angle0_deg = angle_offset[..., 0]
    angle1_deg = angle_offset[..., 0] + angle_offset[..., 1]

    low = tx.np.minimum(angle0_deg, angle1_deg)
    high = tx.np.maximum(angle0_deg, angle1_deg)
    check = (high - low) % 360

    def f(
        ray: tx.Ray,
    ) -> Tuple[Float[Array, "#A #B 2"], Bool[Array, "#A #B 2"]]:
        # print(ray.v, ray.pt)
        length = tx.length(ray.v)
        d, mask = tx.ray_circle_intersection(
            ray.pt, ray.v, 1 + OFFSET * length
        )
        # 2 #A 1

        ang = tx.angle(((d[..., None, None]) * ray.v + ray.pt))
        # 2 #A # B

        mask = mask & (((ang - low) % 360) <= check)
        # #B
        ret = d.transpose(1, 2, 0)
        return ret, mask.transpose(1, 2, 0)
        # 2 #A #B

    return f


def arc_seg(q: V2_t, height: tx.Floating) -> Trail:
    return arc_between_trail(q, tx.ftos(height))


def arc_seg_angle(angle: tx.Floating, dangle: tx.Floating) -> Trail:
    arc_p = tx.to_point(tx.polar(angle))
    return Segment(
        tx.translation(-arc_p), tx.np.array([angle, dangle], float)
    ).to_trail()


def arc_between_trail(q: P2_t, height: tx.Scalars) -> Trail:
    return arc_between(tx.P2(0, 0), q, height).to_trail()
