import math

from dataclasses import dataclass
from typing import Any

from diagrams.bounding_box import BoundingBox
from diagrams.point import Point, ORIGIN


PyCairoContext = Any


@dataclass
class Shape:
    def get_bounding_box(self) -> BoundingBox:
        pass

    def render(self, ctx: PyCairoContext) -> None:
        pass


class Circle(Shape):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius
        self.origin = ORIGIN

    def get_bounding_box(self) -> BoundingBox:
        tl = Point(-self.radius, -self.radius)
        br = Point(+self.radius, +self.radius)
        return BoundingBox(tl, br)

    def render(self, ctx: PyCairoContext) -> None:
        ctx.arc(self.origin.x, self.origin.y, self.radius, 0, 2 * math.pi)


class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        super().__init__()
        self.width = width
        self.height = height
        self.origin = ORIGIN

    def get_bounding_box(self) -> BoundingBox:
        left = self.origin.x - self.width / 2
        top = self.origin.y - self.height / 2
        tl = Point(left, top)
        br = Point(left + self.width, top + self.height)
        return BoundingBox(tl, br)

    def render(self, ctx: PyCairoContext) -> None:
        left = self.origin.x - self.width / 2
        top = self.origin.y - self.height / 2
        ctx.rectangle(left, top, self.width, self.height)
