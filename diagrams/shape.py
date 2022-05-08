import math

from dataclasses import dataclass
from typing import Any, List, Optional

import cairo

from diagrams.bounding_box import BoundingBox
from diagrams.point import Point, ORIGIN


PyCairoContext = Any


@dataclass
class Shape:
    def get_bounding_box(self) -> BoundingBox:
        pass

    def render(self, ctx: PyCairoContext) -> None:
        pass


@dataclass
class Circle(Shape):
    radius: float

    def get_bounding_box(self) -> BoundingBox:
        tl = Point(-self.radius, -self.radius)
        br = Point(+self.radius, +self.radius)
        return BoundingBox(tl, br)

    def render(self, ctx: PyCairoContext) -> None:
        ctx.arc(ORIGIN.x, ORIGIN.y, self.radius, 0, 2 * math.pi)


@dataclass
class Rectangle(Shape):
    width: float
    height: float

    def get_bounding_box(self) -> BoundingBox:
        left = ORIGIN.x - self.width / 2
        top = ORIGIN.y - self.height / 2
        tl = Point(left, top)
        br = Point(left + self.width, top + self.height)
        return BoundingBox(tl, br)

    def render(self, ctx: PyCairoContext) -> None:
        left = ORIGIN.x - self.width / 2
        top = ORIGIN.y - self.height / 2
        ctx.rectangle(left, top, self.width, self.height)


@dataclass
class RoundedRectangle(Shape):
    width: float
    height: float
    radius: float

    def get_bounding_box(self) -> BoundingBox:
        left = ORIGIN.x - self.width / 2
        top = ORIGIN.y - self.height / 2
        tl = Point(left, top)
        br = Point(left + self.width, top + self.height)
        return BoundingBox(tl, br)

    def render(self, ctx: PyCairoContext) -> None:
        x = ORIGIN.x - self.width / 2
        y = ORIGIN.y - self.height / 2
        r = self.radius
        ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
        ctx.arc(x + self.width - r, y + r, r, 3 * math.pi / 2, 0)
        ctx.arc(x + self.width - r, y + self.height - r, r, 0, math.pi / 2)
        ctx.arc(x + r, y + self.height - r, r, math.pi / 2, math.pi)
        ctx.close_path()


@dataclass
class Path(Shape):
    points: List[Point]

    def get_bounding_box(self) -> BoundingBox:
        box = BoundingBox.empty()
        for p in self.points:
            box = box.enclose(p)
        return box

    def render(self, ctx: PyCairoContext) -> None:
        p, *rest = self.points
        ctx.move_to(p.x, p.y)
        for p in rest:
            ctx.line_to(p.x, p.y)


@dataclass
class Text(Shape):
    text: str
    font_size: Optional[float]

    def __post_init__(self) -> None:
        surface = cairo.RecordingSurface(
            cairo.Content.COLOR, None # type: ignore
        )
        self.ctx = cairo.Context(surface)

    def get_bounding_box(self) -> BoundingBox:
        self.ctx.select_font_face("sans-serif")
        if self.font_size is not None:
            self.ctx.set_font_size(self.font_size)
        extents = self.ctx.text_extents(self.text)
        left = extents.x_bearing - (extents.width / 2)
        top = extents.y_bearing
        tl = Point(left, top)
        br = Point(left + extents.width, top + extents.height)
        return BoundingBox(tl, br)

    def render(self, ctx: PyCairoContext) -> None:
        ctx.select_font_face("sans-serif")
        if self.font_size is not None:
            ctx.set_font_size(self.font_size)
        extents = ctx.text_extents(self.text)
        ctx.move_to(-(extents.width / 2), (extents.height / 2))
        ctx.text_path(self.text)
