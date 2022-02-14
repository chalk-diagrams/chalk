from dataclasses import dataclass

from diagrams import transform as tx

@dataclass
class Point:
    x: float
    y: float

    def apply_transform(self, t: tx.Transform) -> "Point":
        # see documentation from PyCairo
        # https://pycairo.readthedocs.io/en/latest/reference/matrix.html
        xx, yx, xy, yy, x0, y0 = t()
        new_x = xx * self.x + xy * self.y + x0
        new_y = yx * self.x + yy * self.y + y0
        return Point(new_x, new_y)


ORIGIN = Point(0, 0)
