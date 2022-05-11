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

    def __add__(self, v: "Vector") -> "Point":
        return Point(self.x + v.dx, self.y + v.dy)


@dataclass
class Vector:
    dx: float
    dy: float

    @property
    def length(self):
        return math.sqrt(self.dx ** 2 + self.dy ** 2)

    @property
    def angle(self):
        return math.atan2(self.dy, self.dx)

    @classmethod
    def from_polar(cls, r, angle: float) -> "Vector":
        dx = r * math.cos(angle)
        dy = r * math.sin(angle)
        return cls(dx, dy)

    def rotate(self, by: float) -> "Vector":
        return Vector.from_polar(self.length, self.angle + by)

    def __mul__(self, α: float) -> "Vector":
        return Vector(α * self.dx, α * self.dy)

    __rmul__ = __mul__


ORIGIN = Point(0, 0)
