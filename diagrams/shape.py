import math

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

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
        ctx.arc(x + r, y + r, r, math.pi, 3*math.pi/2)
        ctx.arc(x + self.width-r, y+r, r, 3*math.pi/2, 0)
        ctx.arc(x + self.width-r, y + self.height-r, r, 0, math.pi/2)
        ctx.arc(x+r, y + self.height-r, r, math.pi/2, math.pi)
        ctx.close_path()
        

class PathElement:
    pass


@dataclass
class MoveTo(PathElement):
    point: Point

    @property
    def points(self):
        return [self.point]

    def render(self, ctx: PyCairoContext) -> None:
        ctx.move_to(self.point.x, self.point.y)


@dataclass
class LineTo(PathElement):
    point: Point

    @property
    def points(self):
        return [self.point]

    def render(self, ctx: PyCairoContext) -> None:
        ctx.line_to(self.point.x, self.point.y)


@dataclass
class BezierCurveTo(PathElement):
    cp1: Point
    cp2: Point
    point: Point

    @property
    def points(self):
        return [self.cp1, self.cp2, self.point]

    def render(self, ctx: PyCairoContext) -> None:
        ctx.curve_to(
            self.cp1.x,
            self.cp1.y,
            self.cp2.x,
            self.cp2.y,
            self.point.x,
            self.point.y,
        )


@dataclass
class Path(Shape):
    elements: List[PathElement]

    def get_bounding_box(self) -> BoundingBox:
        box = BoundingBox.empty()
        for elem in self.elements:
            for point in elem.points:
                box = box.enclose(point)
        return box

    def render(self, ctx: PyCairoContext) -> None:
        for elem in self.elements:
            elem.render(ctx)


@dataclass
class Arc(Shape):
    radius: float
    angle0: float
    angle1: float

    def __post_init__(self) -> None:
        surface = cairo.SVGSurface("undefined.svg", 1280, 200)
        self.ctx = cairo.Context(surface)

    def get_bounding_box(self) -> BoundingBox:
        self.render(self.ctx)
        l, t, r, b = self.ctx.path_extents()
        return BoundingBox(Point(l, t), Point(r, b))

    def render(self, ctx: PyCairoContext) -> None:
        ctx.arc(0, 0, self.radius, self.angle0, self.angle1)


@dataclass
class Text(Shape):
    text: str
    font_size: Optional[float]

    def __post_init__(self) -> None:
        surface = cairo.RecordingSurface(cairo.Content.COLOR, None)  # type: ignore
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
        ctx.move_to(-(extents.width / 2),
                    (extents.height / 2))
        ctx.text_path(self.text)
