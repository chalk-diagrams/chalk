import math

from dataclasses import dataclass
from typing import Any, List, Optional

import cairo
import PIL
from io import BytesIO
import cairosvg

from chalk.bounding_box import BoundingBox
from chalk.point import Point, ORIGIN

from svgwrite import Drawing
from svgwrite.base import BaseElement

PyCairoContext = Any


@dataclass
class Shape:
    def get_bounding_box(self) -> BoundingBox:
        pass

    def render(self, ctx: PyCairoContext) -> None:
        pass

    def render_svg(self, dwg: Drawing) -> BaseElement:
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

    def render_svg(self, dwg: Drawing) -> BaseElement:
        return dwg.circle((ORIGIN.x, ORIGIN.y), self.radius)


@dataclass
class Rectangle(Shape):
    width: float
    height: float
    radius: Optional[float] = None

    def get_bounding_box(self) -> BoundingBox:
        left = ORIGIN.x - self.width / 2
        top = ORIGIN.y - self.height / 2
        tl = Point(left, top)
        br = Point(left + self.width, top + self.height)
        return BoundingBox(tl, br)

    def render(self, ctx: PyCairoContext) -> None:
        x = left = ORIGIN.x - self.width / 2
        y = top = ORIGIN.y - self.height / 2
        if self.radius is None:
            ctx.rectangle(left, top, self.width, self.height)
        else:
            r = self.radius
            ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
            ctx.arc(x + self.width - r, y + r, r, 3 * math.pi / 2, 0)
            ctx.arc(x + self.width - r, y + self.height - r, r, 0, math.pi / 2)
            ctx.arc(x + r, y + self.height - r, r, math.pi / 2, math.pi)
            ctx.close_path()

    def render_svg(self, dwg: Drawing) -> BaseElement:
        left = ORIGIN.x - self.width / 2
        top = ORIGIN.y - self.height / 2
        return dwg.rect(
            (left, top),
            (self.width, self.height),
            rx=self.radius,
            ry=self.radius,
        )


@dataclass
class Path(Shape):
    points: List[Point]
    arrow: bool = False

    def get_bounding_box(self) -> BoundingBox:
        box = BoundingBox(self.points[0], self.points[0])
        for p in self.points:
            box = box.enclose(p)
        return box

    def render(self, ctx: PyCairoContext) -> None:
        p, *rest = self.points
        ctx.move_to(p.x, p.y)
        for p in rest:
            ctx.line_to(p.x, p.y)

    def render_svg(self, dwg: Drawing) -> BaseElement:
        line = dwg.polyline([(p.x, p.y) for p in self.points])
        if self.arrow:
            line.set_markers((None, False, dwg.defs.elements[0]))
        return line


@dataclass
class Text(Shape):
    text: str
    font_size: Optional[float]

    def __post_init__(self) -> None:
        surface = cairo.SVGSurface("undefined.svg", 1280, 200)
        self.ctx = cairo.Context(surface)

    def get_bounding_box(self) -> BoundingBox:
        self.ctx.select_font_face("sans-serif")
        if self.font_size is not None:
            self.ctx.set_font_size(self.font_size)
        extents = self.ctx.text_extents(self.text)
        left = extents.x_bearing - (extents.width / 2)
        top = extents.y_bearing
        tl = Point(left, top)
        br = Point(left + extents.x_advance, top + extents.height)
        self.bb = BoundingBox(tl, br)

        return self.bb

    def render(self, ctx: PyCairoContext) -> None:
        ctx.select_font_face("sans-serif")
        if self.font_size is not None:
            ctx.set_font_size(self.font_size)
        extents = ctx.text_extents(self.text)

        ctx.move_to(-(extents.width / 2), (extents.height / 2))
        ctx.text_path(self.text)

    def render_svg(self, dwg: Drawing) -> BaseElement:
        dx = -(self.bb.width / 2)
        return dwg.text(
            self.text,
            transform=f"translate({dx}, 0)",
            style=f"""text-align:center; dominant-baseline:middle;
                      font-family:sans-serif; font-weight: bold;
                      font-size:{self.font_size}px""",
        )


def from_pil(
    im: PIL.Image,
    alpha: float = 1.0,
    format: cairo.Format = cairo.FORMAT_ARGB32,
) -> cairo.Surface:
    assert format in (cairo.FORMAT_RGB24, cairo.FORMAT_ARGB32), (
        "Unsupported pixel format: %s" % format
    )
    if "A" not in im.getbands():
        im.putalpha(int(alpha * 256.0))
    arr = bytearray(im.tobytes("raw", "BGRa"))
    surface = cairo.ImageSurface.create_for_data(
        arr, format, im.width, im.height  # type:ignore
    )
    return surface


@dataclass
class Image(Shape):
    local_path: str
    url_path: Optional[str]

    def __post_init__(self) -> None:
        if self.local_path.endswith("svg"):
            out = BytesIO()
            cairosvg.svg2png(url=self.local_path, write_to=out)
        else:
            out = open(self.local_path, "rb")  # type:ignore

        self.im = PIL.Image.open(out)
        self.height = self.im.height
        self.width = self.im.width

    def get_bounding_box(self) -> BoundingBox:
        left = ORIGIN.x - self.width / 2
        top = ORIGIN.y - self.height / 2
        tl = Point(left, top)
        br = Point(left + self.width, top + self.height)
        return BoundingBox(tl, br)

    def render(self, ctx: PyCairoContext) -> None:
        surface = from_pil(self.im)
        ctx.set_source_surface(surface, -(self.width / 2), -(self.height / 2))
        ctx.paint()

    def render_svg(self, dwg: Drawing) -> BaseElement:
        dx = -self.width / 2
        dy = -self.height / 2
        return dwg.image(
            href=self.url_path, transform=f"translate({dx}, {dy})"
        )
