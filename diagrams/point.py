from dataclasses import dataclass

from diagrams import transform as tx

@dataclass
class Point:
    x: float
    y: float

    def apply_transform(self, t: tx.Transform) -> "Point":
        new_x, new_y = t().transform_point(self.x, self.y)
        return Point(new_x, new_y)


ORIGIN = Point(0, 0)
