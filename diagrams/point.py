import math

from dataclasses import dataclass

from diagrams import transform as tx


@dataclass
class Point:
    x: float
    y: float

    def apply_transform(self, t: tx.Transform) -> "Point":
        new_x, new_y = t().transform_point(self.x, self.y)
        return Point(new_x, new_y)

    def __sub__(self, other: "Point") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y)


@dataclass
class Vector:
    dx: float
    dy: float

    def norm(self):
        return math.sqrt(self.dx ** 2 + self.dy ** 2)

    def angle(self):
        return math.atan2(self.dy, self.dx)


ORIGIN = Point(0, 0)
