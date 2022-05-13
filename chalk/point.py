import math

from dataclasses import dataclass

from chalk import transform as tx


@dataclass
class Point(tx.Transformable):
    x: float
    y: float

    def apply_transform(self, t: tx.Transform):  # type: ignore
        new_x, new_y = t().transform_point(self.x, self.y)
        return Point(new_x, new_y)

    def __add__(self, other: "Vector") -> "Point":
        return Point(self.x + other.dx, self.y + other.dy)

    def __sub__(self, other: "Point") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y)


@dataclass
class Vector(tx.Transformable):
    dx: float
    dy: float

    @property
    def length(self) -> float:
        return math.sqrt(self.dx ** 2 + self.dy ** 2)

    @property
    def angle(self) -> float:
        return math.atan2(self.dy, self.dx)

    @classmethod
    def from_polar(cls, r: float, angle: float) -> "Vector":
        dx = r * math.cos(angle)
        dy = r * math.sin(angle)
        return cls(dx, dy)

    def apply_transform(self, t: tx.Transform):  # type:ignore
        new_dx, new_dy = t().transform_point(self.dx, self.dy)
        return Vector(new_dx, new_dy)

    def rotate(self, by: float) -> "Vector":
        return Vector.from_polar(self.length, self.angle + by)

    def __mul__(self, α: float) -> "Vector":
        return Vector(α * self.dx, α * self.dy)

    __rmul__ = __mul__

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.dx + other.dx, self.dy + other.dy)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.dx - other.dx, self.dy - other.dy)

    def __neg__(self) -> "Vector":
        return Vector(-self.dx, -self.dy)


ORIGIN = Point(0, 0)
