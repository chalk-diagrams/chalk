from diagrams.core import Primitive
from diagrams.point import Point
from diagrams.shape import Path
from diagrams import transform as tx
from typing import List, Any

class Trail:
    def __init__(self, offsets : List[Point]):
        self.offsets = offsets

    def __add__(self, other: "Trail") -> "Trail":
        return Trail(self.offsets + other.offsets)

    def transform(self, t: tx.Transform) -> "Trail":
        return Trail([p.apply_transform(t) for p in self.offsets])

    @staticmethod
    def from_path(path : Any) -> "Trail":
        pts = path.shape.points
        offsets = [t - s for s, t in zip(pts, pts[1:])]
        return Trail(offsets)

    def stroke(self) -> Primitive:
        points = [Point(0, 0)]
        for s in self.offsets:
            points.append(s + points[-1])
        return Primitive.from_shape(Path(points))
