from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import chalk.shapes.arc as arc
import chalk.transform as tx
from chalk.envelope import Envelope
from chalk.monoid import Monoid
from chalk.shapes.arc import Segment
from chalk.trace import Trace
from chalk.transform import Affine, Floating, P2_t, Transformable, V2_t
from chalk.types import Diagram, Enveloped, Traceable, TrailLike

if TYPE_CHECKING:

    from chalk.shapes.path import Path


@dataclass
class Located(Enveloped, Traceable, Transformable):
    trail: Trail
    location: P2_t

    def split(self, i: int) -> Located:
        return Located(self.trail.split(i), self.location[i])

    def located_segments(self) -> Segment:
        pts = self.points()
        return self.trail.segments.apply_transform(tx.translation(pts))

    def points(self) -> P2_t:
        return self.trail.points() + self.location

    def get_envelope(self) -> Envelope:
        s = self.located_segments()
        t = s.transform
        return Envelope.general_transform(t, lambda x: arc.arc_envelope(s, x))

    def get_trace(self) -> Trace:
        s = self.located_segments()
        t = s.transform
        return Trace.general_transform(t, lambda x: arc.arc_trace(s, x))

    def stroke(self) -> Diagram:
        return self.to_path().stroke()

    def apply_transform(self, t: Affine) -> Located:
        return Located(self.trail.apply_transform(tx.remove_translation(t)), 
                       t @ self.location)

    def to_path(self) -> Path:
        from chalk.shapes.path import Path

        return Path([self])


@dataclass
class Trail(Monoid, Transformable, TrailLike):
    segments: Segment

    closed: tx.Mask = field(default_factory=lambda: tx.X.np.array(False))

    def split(self, i: int) -> Trail:
        return Trail(self.segments.split(i), self.closed[i])

    # Monoid
    @staticmethod
    def empty() -> Trail:
        return Trail(
            Segment(tx.X.np.zeros((0, 3, 3)), tx.X.np.zeros((0, 2))),
            tx.X.np.array(False),
        )

    def __add__(self, other: Trail) -> Trail:
        # assert not (self.closed or other.closed), "Cannot add closed trails"
        return Trail(self.segments + other.segments, tx.X.np.array(False))

    # Transformable
    def apply_transform(self, t: Affine) -> Trail:
        t = tx.remove_translation(t)
        return Trail(self.segments.apply_transform(t), self.closed)

    # Trail-like
    def to_trail(self) -> Trail:
        return self

    def close(self) -> Trail:
        return Trail(self.segments, tx.X.np.array(True))

    def points(self) -> P2_t:
        q = self.segments.q
        return tx.to_point(tx.X.np.cumsum(q, axis=-3) - q)

    def at(self, p: P2_t) -> Located:
        return Located(self, p)

    # def reverse(self) -> Trail:
    #     return Trail(
    #         [seg.reverse() for seg in reversed(self.segments)],
    #         reversed(segment_angles),
    #         self.closed,
    #     )

    def centered(self) -> Located:
        return self.at(
            -sum(self.points(), tx.P2(0, 0)) / self.segments.t.shape[0]
        )

    # # Misc. Constructor
    @staticmethod
    def from_offsets(offsets: List[V2_t], closed: bool = False) -> Trail:
        trail = Trail.concat([arc.seg(off) for off in offsets])
        if closed:
            trail = trail.close()
        return trail

    @staticmethod
    def hrule(length: Floating) -> Trail:
        return arc.seg(length * tx.X.unit_x)

    @staticmethod
    def vrule(length: Floating) -> Trail:
        return arc.seg(length * tx.X.unit_y)

    @staticmethod
    def rectangle(width: Floating, height: Floating) -> Trail:
        t = arc.seg(tx.X.unit_x) + arc.seg(tx.X.unit_y)
        return (t + t.rotate_by(0.5)).close().scale_x(width).scale_y(height)

    @staticmethod
    def rounded_rectangle(
        width: Floating, height: Floating, radius: Floating
    ) -> Trail:
        r = radius
        edge1 = math.sqrt(2 * r * r) / 2
        edge3 = math.sqrt(r * r - edge1 * edge1)
        corner = arc.arc_seg(tx.V2(r, r), -(r - edge3))
        b = [height - r, width - r, height - r, width - r]
        trail = Trail.concat(
            (arc.seg(b[i] * tx.X.unit_y) + corner).rotate_by(i / 4)
            for i in range(4)
        ) + arc.seg(0.01 * tx.X.unit_y)
        return trail.close()

    @staticmethod
    def circle(radius: Floating = 1.0, clockwise: bool = True) -> Trail:
        sides = 4
        dangle = -90
        rotate_by = 1
        if not clockwise:
            dangle = 90
            rotate_by *= -1
        return (
            Trail.concat(
                [
                    arc.arc_seg_angle(0, dangle).rotate_by(
                        rotate_by * i / sides
                    )
                    for i in range(sides)
                ]
            )
            .close()
            .scale(radius)
        )

    @staticmethod
    def regular_polygon(sides: int, side_length: Floating) -> Trail:
        edge = Trail.hrule(1)
        return (
            Trail.concat(edge.rotate_by(i / sides) for i in range(sides))
            .close()
            .scale(side_length)
        )
