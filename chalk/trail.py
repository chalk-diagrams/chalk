import math

from typing import List, Any

from chalk.core import Primitive
from chalk.point import Point, Vector
from chalk.shape import Path
from chalk import transform as tx


class Trail:
    def __init__(self, offsets: List[Vector]):
        self.offsets = offsets

    def __add__(self, other: "Trail") -> "Trail":
        return Trail(self.offsets + other.offsets)

    def transform(self, t: tx.Transform) -> "Trail":
        return Trail([p.apply_transform(t) for p in self.offsets])

    @staticmethod
    def from_path(path: Any) -> "Trail":
        pts = path.shape.points
        offsets = [t - s for s, t in zip(pts, pts[1:])]
        return Trail(offsets)

    def stroke(self) -> Primitive:
        points = [Point(0, 0)]
        for s in self.offsets:
            points.append(points[-1] + s)
        return Primitive.from_shape(Path(points))
