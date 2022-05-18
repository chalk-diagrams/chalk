from typing import List

from chalk.core import Primitive
from chalk.point import Point, Vector, ORIGIN
from chalk.shape import Path
from chalk import transform as tx


class Trail(tx.Transformable):
    def __init__(self, offsets: List[Vector]):
        self.offsets = offsets

    def __add__(self, other: "Trail") -> "Trail":
        return Trail(self.offsets + other.offsets)

    @classmethod
    def from_path(cls, path: Path) -> "Trail":
        pts = path.points
        offsets = [t - s for s, t in zip(pts, pts[1:])]
        return cls(offsets)

    def to_path(self, origin: Point = ORIGIN) -> Path:
        points = [origin]
        for s in self.offsets:
            points.append(points[-1] + s)
        return Path(points)

    def stroke(self) -> Primitive:
        return Primitive.from_shape(self.to_path())

    def transform(self, t: tx.Transform) -> "Trail":
        return Trail([p.apply_transform(t) for p in self.offsets])

    def apply_transform(self, t: tx.Transform) -> "Trail":  # type: ignore
        return self.transform(t)


unit_x = Trail([Vector(1, 0)])
unit_y = Trail([Vector(0, 1)])
