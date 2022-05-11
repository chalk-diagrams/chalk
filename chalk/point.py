from dataclasses import dataclass

from chalk import transform as tx


@dataclass
class Point:
    x: float
    y: float

    def apply_transform(self, t: tx.Transform) -> "Point":
        new_x, new_y = t().transform_point(self.x, self.y)
        return Point(new_x, new_y)

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)


ORIGIN = Point(0, 0)
