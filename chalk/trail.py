from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Iterable, List, Tuple, Union

from chalk.envelope import Envelope
from chalk.shapes.arc import ArcSegment, arc_seg, arc_seg_angle
from chalk.shapes.segment import Segment, seg
from chalk.trace import Trace
from chalk.transform import (
    P2,
    V2,
    Affine,
    Transformable,
    apply_affine,
    remove_translation,
    unit_x,
    unit_y,
)
from chalk.types import Diagram, Enveloped, Traceable, TrailLike

if TYPE_CHECKING:
    from chalk.shapes.path import Path

SegmentLike = Union[Segment, ArcSegment]


@dataclass
class Located(Enveloped, Traceable, Transformable):
    trail: Trail
    location: P2

    def located_segments(self) -> Iterable[Tuple[SegmentLike, P2]]:
        return zip(self.trail.segments, self.points())

    def points(self) -> Iterable[P2]:
        return (pt + self.location for pt in self.trail.points())

    def get_envelope(self) -> Envelope:
        return Envelope.concat(
            segment.get_envelope().translate_by(location)
            for segment, location in self.located_segments()
        )

    def get_trace(self) -> Trace:
        return Trace.concat(
            segment.get_trace().translate_by(location)
            for segment, location in self.located_segments()
        )

    def stroke(self) -> Diagram:
        return self.to_path().stroke()

    def apply_transform(self, t: Affine) -> Located:  # type: ignore
        return Located(
            apply_affine(t, self.trail), apply_affine(t, self.location)
        )

    def to_path(self) -> Path:
        from chalk.shapes.path import Path

        return Path([self])


@dataclass
class Trail(Transformable, TrailLike):
    segments: List[SegmentLike]
    closed: bool = False

    # Monoid - concat
    @staticmethod
    def empty() -> Trail:
        return Trail([], False)

    def __add__(self, other: Trail) -> Trail:
        assert not (self.closed or other.closed), "Cannot add closed trails"
        return Trail(self.segments + other.segments, False)

    @staticmethod
    def concat(trails: Iterable[Trail]) -> Trail:
        return reduce(Trail.__add__, trails, Trail.empty())

    def close(self) -> Trail:
        return Trail(self.segments, True)

    def points(self) -> Iterable[P2]:
        cur = P2(0, 0)
        pts = [cur]
        for segment in self.segments:
            cur += segment.q
            pts.append(cur)
        return pts

    # Apply transform to all
    def apply_transform(self, t: Affine) -> Trail:  # type: ignore
        t = remove_translation(t)
        return Trail(
            [seg.apply_transform(t) for seg in self.segments], self.closed
        )

    def at(self, p: P2) -> Located:
        return Located(self, p)

    def reverse(self) -> Trail:
        return Trail(
            [seg.scale(-1) for seg in reversed(self.segments)],
            self.closed,
        )

    # Constructor
    @staticmethod
    def from_offsets(offsets: List[V2], closed: bool = False) -> Trail:
        return Trail([Segment(off) for off in offsets], closed)

    @staticmethod
    def hrule(length: float) -> Trail:
        return seg(length * unit_x)

    @staticmethod
    def vrule(length: float) -> Trail:
        return seg(length * unit_y)

    @staticmethod
    def rectangle(width: float, height: float) -> Trail:
        t = seg(unit_x * width) + seg(unit_y * height)
        return (t + t.rotate_by(0.5)).close()

    @staticmethod
    def rounded_rectangle(width: float, height: float, radius: float) -> Trail:
        r = radius
        edge1 = math.sqrt(2 * r * r) / 2
        edge3 = math.sqrt(r * r - edge1 * edge1)
        corner = arc_seg(V2(r, r), -(r - edge3))
        b = [height - r, width - r, height - r, width - r]
        trail = Trail.concat(
            (seg(b[i] * unit_y) + corner).rotate_by(i / 4) for i in range(4)
        ) + seg(0.01 * unit_y)
        return trail.close()

    def centered(self) -> Located:
        return self.at(-sum(self.points(), P2(0, 0)) / len(self.segments))

    @staticmethod
    def circle(radius: float = 1.0, clockwise: bool = True) -> Trail:
        sides = 4
        dangle = -90
        rotate_by = 1
        if not clockwise:
            dangle = 90
            rotate_by *= -1
        return (
            Trail.concat(
                [
                    arc_seg_angle(0, dangle).rotate_by(rotate_by * i / sides)
                    for i in range(sides)
                ]
            )
            .close()
            .scale(radius)
        )

    # @staticmethod
    # def polygon(sides: int) -> Path:
    #     edge = Trail.hrule(1.0)
    #     return Trail.concat(edge.rotate_by(i / side) for i in range(sides))

    @staticmethod
    def regular_polygon(sides: int, side_length: float) -> Trail:
        edge = Trail.hrule(side_length)
        return Trail.concat(
            edge.rotate_by(i / sides) for i in range(sides)
        ).close()


# unit_x = Trail.from_offsets([V2(1, 0)])
# unit_y = Trail.from_offsets([V2(0, 1)])
