from dataclasses import dataclass

from diagrams.point import Point, ORIGIN
from diagrams.transform import Transform


@dataclass
class BoundingBox:
    tl: Point
    br: Point

    @classmethod
    def from_limits(cls, left: float, top: float, right: float, bottom: float) -> "BoundingBox":
        tl = Point(left, top)
        br = Point(right, bottom)
        return cls(tl, br)

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

    def transform(self, t: "Transform") -> "BoundingBox":
        # TODO
        print("TODO · BoundingBox.transform")
        return self

    def union(self, other: "BoundingBox") -> "BoundingBox":
        left = min(self.tl.x, other.tl.x)
        top = min(self.tl.y, other.tl.y)
        right = max(self.br.x, other.br.x)
        bottom = max(self.br.y, other.br.y)
        return BoundingBox.from_limits(left, top, right, bottom)


EMPTY_BOUNDING_BOX = BoundingBox(ORIGIN, ORIGIN)
