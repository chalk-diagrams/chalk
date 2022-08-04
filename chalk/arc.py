"""
Contains arithmetic for arc calculations.
"""


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, NewType

from planar.py import Ray

import chalk.transform as tx
from chalk.envelope import Envelope
from chalk.trace import Trace
from chalk.transform import P2, V2, unit_x, unit_y, to_radians, from_radians


Degrees = NewType('Degrees', float)

def is_in_mod_360(x: Degrees, a: Degrees, b: Degrees) -> bool:
    """Checks if x ∈ [a, b] mod 360. See the following link for an
    explanation:
    https://fgiesen.wordpress.com/2015/09/24/intervals-in-modular-arithmetic/
    """
    return (x - a) % 360 <= (b - a) % 360

# Several ways to specify an arc. Needed for different backends

def angle(center: P2, point: P2) -> Degrees:
    return Degrees((point - center).angle)

def point(center: P2, radius: float, angle: Degrees) -> P2:
    return P2.polar(angle, radius) + center

def mid(p1: P2, p2 : P2) -> P2:
    return p1 + (p2 - p1) / 2 

@dataclass
class ArcByDegrees:
    "Specify arc by angles from center"
    angle0 : Degrees
    angle1 : Degrees
    radius : float
    center : P2

    def to_arc_between(self) -> ArcBetween:
        point0 = point(self.center, self.radius, self.angle0)
        point1 = point(self.center, self.radius, self.angle1)
        point_angle_mid = point(self.center, self.radius, (self.angle0 - self.angle1) / 2)
        point_mid = mid(point0, point1)
        height = (point_angle_mid - point_mid).length
        # if point_angle_mid.y > point_mid.y:
        #     height = -height
        return ArcBetween(point0, point1, height)
    
@dataclass
class ArcBetween:
    "Specify arc by two points and a height"
    point0 : P2
    point1 : P2
    height : float

    def radius(self):
        # solve triangle r^2 = (r- h)^2 + (d)^2
        d = (self.point1 - self.point0).length / 2
        return self.height / 2 + (d * d) / (2 * self.height)
    
    def to_arc_by_degrees(self) -> ArcByDegrees:
        flip = 1 if self.height > 0 else -1
        r = self.radius()
        diff = (self.point1 - self.point0).perpendicular()
        h_v = flip * diff.normalized()
        mid_p = mid(self.point0, self.point1)
        center = mid_p - h_v * (r - abs(self.height))
        angle0 = angle(center, self.point0)
        angle1 = angle(center, self.point1)
        return ArcByDegrees(angle0, angle1, r, center)

    def to_arc_radius(self) -> ArcRadius:
        return ArcRadius(self.point0, self.point1, 0, self.radius())
    
@dataclass
class ArcRadius:
    "Specify arc by two points and a radius"
    point0 : P2
    point1 : P2
    large : bool
    radius : float

    def height(self):
        d = (self.point1 - self.point0).length / 2
        # solve triangle r^2 = (r- h)^2 + d / 2
        r = self.radius
        if self.large:
            return r + math.sqrt(r * r - d * d)
        else:
            return r - math.sqrt(r * r - d * d)
        
    
    def to_arc_between(self) -> ArcByDegrees:
        return ArcBetween(self.point0, self.point1, self.height())


@dataclass
class ArcSegment(tx.Transformable):
    angle0: float
    angle1: float
    t: tx.Transform = tx.Affine.identity()

    @property
    def _start(self) -> P2:
        return P2(math.cos(to_radians(self.angle0)), math.sin(to_radians(self.angle0)))

    @property
    def _end(self) -> P2:
        return P2(math.cos(to_radians(self.angle1)), math.sin(to_radians(self.angle1)))


    @staticmethod
    def arc_between(p: P2, q: P2, height: float):
        h = abs(height)
        d = (q - p).length
        # Determine the arc's angle θ and its radius r
        θ = math.acos((d**2 - 4.0 * h**2) / (d**2 + 4.0 * h**2))
        r = d / (2 * math.sin(θ))

        if height > 0:
            # bend left
            φ = -math.pi / 2
            dy = r - h
        else:
            # bend right
            φ = +math.pi / 2
            dy = h - r

        diff = q - p
        r = (
            ArcSegment(-from_radians(θ), from_radians(θ))
            .scale(r)
            .rotate_rad(math.pi / 2)
            .translate(d/2, dy))
        if height > 0:
            r  = r.reflect_y()
        r = (r.rotate(-diff.angle)
            .translate_by(p)
        )
        return r
    
    
    @staticmethod
    def ellipse_between(p: P2, q: P2, height: float):
        diff = q - p
        r = (
            ArcSegment(180, 360)
            .scale_y(height)
            .translate_by(unit_x)
            .scale_x(diff.length / 2)
            .rotate(-diff.angle)
            .translate_by(p)
        )
        return r

    @property
    def p(self):
        return tx.apply_affine(self.t, self._start)

    @property
    def q(self):
        return tx.apply_affine(self.t, self._end)

    def apply_transform(self, t: tx.Affine) -> ArcSegment:
        return ArcSegment(self.angle0, self.angle1, t * self.t)

    def get_trace(self) -> Trace:
        angle0_deg = self.angle0
        angle1_deg = self.angle1

        def f(p: P2, v: V2) -> List[SignedDistance]:
            ray = Ray(p, v)
            # Same as circle but check that angle is in arc.
            return sorted(
                [
                    d / v.length
                    for d in ray_circle_intersection(ray, 1)
                    if is_in_mod_360(
                        ((d * v) + self._start).angle, angle0_deg, angle1_deg
                    )
                ]
            )

        return Trace(f).apply_transform(self.t)

    def get_envelope(self) -> Envelope:
        angle0_deg = self.angle0
        angle1_deg = self.angle1

        v1 = V2.polar(angle0_deg, 1)
        v2 = V2.polar(angle1_deg, 1)

        def wrapped(d: V2) -> SignedDistance:
            is_circle = abs(angle0_deg - angle1_deg) >= 360
            if is_circle or is_in_mod_360(d.angle, angle0_deg, angle1_deg):
                # Case 1: P2 at arc
                return 1 / d.length  # type: ignore
            else:
                # Case 2: P2 outside of arc
                x: float = max(d.dot(v1), d.dot(v2))
                return x

        return Envelope(wrapped).apply_transform(self.t)

    def render_path(self, ctx):
        end_point = tx.apply_affine(self.t, self._end)
        t2 = tx.remove_translation(self.t)
        r_x = tx.apply_affine(t2, unit_x)
        r_y = tx.apply_affine(t2, unit_y)
        rot = r_x.angle
        start = 180 + rot
        x, y = self.p.x - math.cos(to_radians(start)), self.p.y - math.sin(to_radians(start))
        ctx.new_sub_path()
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(r_x.length, r_y.length)
        ctx.arc(0., 0., 1., start, 180 + rot + self.angle)
        ctx.restore()


    def render_svg_path(self) -> str:
        end_point = tx.apply_affine(self.t, self._end)
        large = 0
        t2 = tx.remove_translation(self.t)
        r_x = tx.apply_affine(t2, unit_x)
        r_y = tx.apply_affine(t2, unit_y)
        rot = r_x.angle
        return f"A {r_x.length} {r_y.length} {rot} {large} 0 {end_point.x} {end_point.y}"

    def render_tikz_path(self, pts, pylatex) -> None:    
        t2 = tx.remove_translation(self.t)
        r_x = tx.apply_affine(t2, unit_x)
        r_y = tx.apply_affine(t2, unit_y)
        rot = r_x.angle
        print(r_x, r_y, self.angle0, self.angle1)
        pts._arg_list.append(pylatex.TikZUserPath(f"{{[rotate={rot}] arc[start angle={self.angle0}, end angle={self.angle1}, x radius={r_x.length}, y radius ={r_y.length}]}}"))
