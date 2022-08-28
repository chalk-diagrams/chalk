from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from chalk import transform as tx
from chalk.envelope import Envelope
from chalk.shapes.shape import Shape
from chalk.trace import Trace
from chalk.trail import Located, Trail
from chalk.transform import P2, Transformable
from chalk.types import Diagram, Enveloped, Traceable
from chalk.visitor import A, ShapeVisitor


def make_path(
    segments: List[Tuple[float, float]], closed: bool = False
) -> Diagram:
    return Path.from_list_of_tuples(segments, closed).stroke()


@dataclass
class Path(Shape, Enveloped, Traceable, Transformable):
    """Path class."""

    loc_trails: List[Located]

    # Monoid - compose
    @staticmethod
    def empty() -> Path:
        return Path([])

    def __add__(self, other: Path) -> Path:
        return Path(self.loc_trails + other.loc_trails)

    def apply_transform(self, t: tx.Affine) -> Path:  # type: ignore
        return Path(
            [loc_trail.apply_transform(t) for loc_trail in self.loc_trails]
        )

    def points(self) -> Iterable[P2]:
        for loc_trails in self.loc_trails:
            for pt in loc_trails.points():
                yield pt

    def get_envelope(self) -> Envelope:
        return Envelope.concat((loc.get_envelope() for loc in self.loc_trails))

    def get_trace(self) -> Trace:
        return Trace.concat((loc.get_trace() for loc in self.loc_trails))

    def accept(self, visitor: ShapeVisitor[A], **kwargs: Any) -> A:
        return visitor.visit_path(self, **kwargs)

    # Constructors
    @staticmethod
    def from_points(points: List[P2], closed: bool = False) -> Path:
        if not points:
            return Path.empty()
        start = points[0]
        trail = Trail.from_offsets(
            [pt2 - pt1 for pt1, pt2 in zip(points, points[1:])], closed
        )
        return Path([trail.at(start)])

    @staticmethod
    def from_point(point: P2) -> Path:
        return Path.from_points([point])

    @staticmethod
    def from_pairs(segs: List[Tuple[P2, P2]], closed: bool = False) -> Path:
        if not segs:
            return Path.empty()
        ls = [segs[0][0]]
        for seg in segs:
            assert seg[0] == ls[-1]
            ls.append(seg[1])
        return Path.from_points(ls, closed)

    @staticmethod
    def from_list_of_tuples(
        coords: List[Tuple[float, float]], closed: bool = False
    ) -> Path:
        points = list([P2(x, y) for x, y in coords])
        return Path.from_points(points, closed)
