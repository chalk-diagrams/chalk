from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from chalk import transform as tx
# from chalk.envelope import Envelope
from chalk.shapes.shape import Shape
from chalk.trace import Trace, TraceDistances
from chalk.trail import Located, Trail
from chalk.transform import P2_t, Transformable
from chalk.types import Diagram, Enveloped, Traceable
from chalk.visitor import C, ShapeVisitor


def make_path(
    segments: List[Tuple[float, float]], closed: bool = False
) -> Diagram:
    return Path.from_list_of_tuples(segments, closed).stroke()


@dataclass(unsafe_hash=True, frozen=True)
class Path(Shape, Enveloped, Traceable, Transformable):
    """Path class."""

    loc_trails: Sequence[Located]

    def split(self, i: int) -> Path:
        return Path(tuple([loc.split(i) for loc in self.loc_trails]))

    # Monoid - compose
    @staticmethod
    def empty() -> Path:
        return Path(())

    def __add__(self, other: Path) -> Path:
        return Path(self.loc_trails + other.loc_trails)

    def apply_transform(self, t: tx.Affine) -> Path:
        return Path(
            tuple([loc_trail.apply_transform(t) for loc_trail in self.loc_trails])
        )

    def points(self) -> Iterable[P2_t]:
        for loc_trails in self.loc_trails:
            for pt in loc_trails.points():
                yield pt

    # def split(self):
    #     return [loc.trail.segments.get(i).at(pt).stroke()
    #             for loc in self.loc_trails
    #             for i, pt in enumerate(loc.points()) ]

    def envelope(self, t) -> Scalars:
        return max((loc.envelope(t) for loc in self.loc_trails))

    def get_trace(self, t) -> Trace:
        return TraceDistances.concat(
            (loc.get_trace()(t) for loc in self.loc_trails))

    def accept(self, visitor: ShapeVisitor[C], **kwargs: Any) -> C:
        return visitor.visit_path(self, **kwargs)

    # Constructors
    @staticmethod
    def from_points(points: List[P2_t], closed: bool = False) -> Path:
        if not points:
            return Path.empty()
        start = points[0]
        trail = Trail.from_offsets(
            [pt2 - pt1 for pt1, pt2 in zip(points, points[1:])], closed
        )
        return Path(tuple([trail.at(start)]))

    @staticmethod
    def from_point(point: P2_t) -> Path:
        return Path.from_points([point])

    @staticmethod
    def from_pairs(
        segs: List[Tuple[P2_t, P2_t]], closed: bool = False
    ) -> Path:
        if not segs:
            return Path.empty()
        ls = [segs[0][0]]
        for seg in segs:
            assert seg[0] == ls[-1]
            ls.append(seg[1])
        return Path.from_points(tuple(ls), closed)

    @staticmethod
    def from_list_of_tuples(
        coords: List[Tuple[float, float]], closed: bool = False
    ) -> Path:
        points = list([tx.P2(x, y) for x, y in coords])
        return Path.from_points(points, closed)
