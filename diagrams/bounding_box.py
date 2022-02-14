from dataclasses import dataclass

from diagrams.point import Point, ORIGIN
from diagrams.transform import Transform


@dataclass
class BoundingBox:
    tl: Point
    br: Point

    @classmethod
    def from_limits(
        cls, left: float, top: float, right: float, bottom: float
    ) -> "BoundingBox":
        tl = Point(left, top)
        br = Point(right, bottom)
        return cls(tl, br)

    @classmethod
    def empty(cls) -> "BoundingBox":
        return cls(ORIGIN, ORIGIN)

    @property
    def width(self) -> float:
        return self.br.x - self.tl.x

    @property
    def height(self) -> float:
        return self.br.y - self.tl.y

    @property
    def left(self) -> float:
        return self.tl.x

    @property
    def top(self) -> float:
        return self.tl.y

    @property
    def right(self) -> float:
        return self.br.x

    @property
    def bottom(self) -> float:
        return self.br.y

    def enclose(self, point: Point) -> "BoundingBox":
        return BoundingBox.from_limits(
            min(self.left, point.x),
            min(self.top, point.y),
            max(self.right, point.x),
            max(self.bottom, point.y),
        )

    def apply_transform(self, t: Transform) -> "BoundingBox":
        return (
            BoundingBox.empty()
            .enclose(Point(self.left, self.top).apply_transform(t))
            .enclose(Point(self.right, self.top).apply_transform(t))
            .enclose(Point(self.left, self.bottom).apply_transform(t))
            .enclose(Point(self.right, self.bottom).apply_transform(t))
        )

    def union(self, other: "BoundingBox") -> "BoundingBox":
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        return BoundingBox.from_limits(left, top, right, bottom)
