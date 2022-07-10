from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO
from typing import Any, List, Optional, Tuple

import PIL
from svgwrite import Drawing
from svgwrite.base import BaseElement
from svgwrite.shapes import Rect

from chalk import transform as tx
from chalk.envelope import Envelope
from chalk.segment import Segment, ray_circle_intersection
from chalk.style import Style
from chalk.trace import SignedDistance, Trace
from chalk.transform import P2, V2, BoundingBox, Ray, Vec2Array, origin

PyLatex = Any
PyLatexElement = Any
PyCairoContext = Any
PyCairoSurface = Any


@dataclass
class Shape:
    """Shape class."""

    def get_bounding_box(self) -> BoundingBox:
        raise NotImplementedError

    def get_envelope(self) -> Envelope:
        return Envelope.from_bounding_box(self.get_bounding_box())

    def get_trace(self) -> Trace:
        # default trace based on bounding box
        box = self.get_bounding_box()
        return Path.rectangle(box.width, box.height).get_trace()

    def render(self, ctx: PyCairoContext) -> None:
        pass

    def render_svg(self, dwg: Drawing) -> BaseElement:
        return dwg.g()

    def render_tikz(self, p: PyLatex, style: Style) -> PyLatexElement:
        return p.TikZScope()


@dataclass
class Circle(Shape):
    """Circle class."""

    radius: float

    def get_envelope(self) -> Envelope:
        return Envelope.from_circle(self.radius)

    def get_trace(self) -> Trace:
        def f(p: P2, v: V2) -> List[SignedDistance]:
            ray = Ray(p, v)
            return sorted(
                [
                    d / v.length
                    for d in ray_circle_intersection(ray, self.radius)
                ]
            )

        return Trace(f)

    def render(self, ctx: PyCairoContext) -> None:
        ctx.arc(origin.x, origin.y, self.radius, 0, 2 * math.pi)

    def render_svg(self, dwg: Drawing) -> BaseElement:
        return dwg.circle(
            (origin.x, origin.y),
            self.radius,
            style="vector-effect: non-scaling-stroke",
        )

    def render_tikz(self, pylatex: PyLatex, style: Style) -> PyLatexElement:
        return pylatex.TikZDraw(
            [pylatex.TikZCoordinate(0, 0), "circle"],
            options=pylatex.TikZOptions(
                radius=self.radius, **style.to_tikz(pylatex)
            ),
        )


@dataclass
class Rectangle(Shape):
    """Rectangle class."""

    width: float
    height: float
    radius: Optional[float] = None

    def get_bounding_box(self) -> BoundingBox:
        left = origin.x - self.width / 2
        top = origin.y - self.height / 2
        tl = P2(left, top)
        br = P2(left + self.width, top + self.height)
        return BoundingBox([tl, br])

    def get_trace(self) -> Trace:
        # FIXME For rounded corners the following trace is not accurate
        return Path.rectangle(self.width, self.height).get_trace()

    def render(self, ctx: PyCairoContext) -> None:
        x = left = origin.x - self.width / 2
        y = top = origin.y - self.height / 2
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
        left = origin.x - self.width / 2
        top = origin.y - self.height / 2
        return dwg.rect(
            (left, top),
            (self.width, self.height),
            rx=self.radius,
            ry=self.radius,
            style="vector-effect: non-scaling-stroke;",
        )

    def render_tikz(self, pylatex: PyLatex, style: Style) -> PyLatexElement:
        left = origin.x - self.width / 2
        top = origin.y - self.height / 2
        return pylatex.TikZDraw(
            [
                pylatex.TikZCoordinate(left, top),
                "rectangle",
                pylatex.TikZCoordinate(left + self.width, top + self.height),
            ],
            options=pylatex.TikZOptions(**style.to_tikz(pylatex)),
        )


@dataclass
class Path(Shape, tx.Transformable):
    """Path class."""

    points: Vec2Array
    arrow: bool = False

    @classmethod
    def from_point(cls, point: P2) -> Path:
        return cls(Vec2Array([point]))

    @classmethod
    def from_points(cls, points: List[P2]) -> Path:
        return cls(Vec2Array(points))

    @classmethod
    def from_list_of_tuples(
        cls, coords: List[Tuple[float, float]], arrow: bool = False
    ) -> Path:
        points = [P2(x, y) for x, y in coords]
        return cls(Vec2Array(points), arrow)

    @property
    def segments(self) -> List[Segment]:
        return [
            Segment(p, q) for p, q in zip(self.points[1:], self.points[:-1])
        ]

    @staticmethod
    def hrule(length: float) -> Path:
        return Path.from_list_of_tuples([(-length / 2, 0), (length / 2, 0)])

    @staticmethod
    def vrule(length: float) -> Path:
        return Path.from_list_of_tuples([(0, -length / 2), (0, length / 2)])

    @staticmethod
    def rectangle(width: float, height: float) -> Path:
        # Should I reuse the `polygon` function to define `rectangle`?
        # polygon(4, 1, math.pi / 4).scale_x(width).scale_y(height)
        x = width / 2
        y = height / 2
        return Path.from_list_of_tuples(
            [(-x, y), (x, y), (x, -y), (-x, -y), (-x, y)]
        )

    @staticmethod
    def polygon(sides: int, radius: float, rotation: float = 0) -> Path:
        coords = []
        n = sides + 1
        for s in range(n):
            # Rotate to align with x axis.
            t = 2.0 * math.pi * s / sides + (math.pi / 2 * sides) + rotation
            coords.append((radius * math.cos(t), radius * math.sin(t)))
        return Path.from_list_of_tuples(coords)

    @staticmethod
    def regular_polygon(sides: int, side_length: float) -> Path:
        return Path.polygon(
            sides, side_length / (2 * math.sin(math.pi / sides))
        )

    def get_envelope(self) -> Envelope:
        return Envelope.from_path(self.points)

    # def get_bounding_box(self) -> BoundingBox:
    #     return BoundingBox.from_points(self.points)

    def get_trace(self) -> Trace:
        return Trace.concat(segment.get_trace() for segment in self.segments)

    def apply_transform(self, t: tx.Affine) -> Path:  # type: ignore
        return Path(tx.apply_affine(t, self.points))

    def render(self, ctx: PyCairoContext) -> None:
        p, *rest = self.points
        ctx.move_to(p.x, p.y)
        for p in rest:
            ctx.line_to(p.x, p.y)

    def render_svg(self, dwg: Drawing) -> BaseElement:
        line = dwg.polyline(
            [(p.x, p.y) for p in self.points],
            style="vector-effect: non-scaling-stroke;",
        )
        if self.arrow:
            line.set_markers((None, False, dwg.defs.elements[0]))
        return line

    def render_tikz(self, pylatex: PyLatex, style: Style) -> PyLatexElement:
        pts = []
        for p in self.points:
            pts.append(pylatex.TikZCoordinate(p.x, p.y))
            pts.append("--")

        return pylatex.TikZDraw(
            pts[:-1], options=pylatex.TikZOptions(**style.to_tikz(pylatex))
        )


def is_in_mod_360(x: float, a: float, b: float) -> bool:
    """Checks if x ∈ [a, b] mod 360. See the following link for an
    explanation:
    https://fgiesen.wordpress.com/2015/09/24/intervals-in-modular-arithmetic/
    """
    return (x - a) % 360 <= (b - a) % 360


@dataclass
class Arc(Shape):
    """Arc class."""

    radius: float
    angle0: float
    angle1: float

    def __post_init__(self) -> None:
        self.angle0, self.angle1 = -self.angle1, -self.angle0

    def get_trace(self) -> Trace:
        angle0_deg = self.angle0 * (180 / math.pi)
        angle1_deg = self.angle1 * (180 / math.pi)

        def f(p: P2, v: V2) -> List[SignedDistance]:
            ray = Ray(p, v)
            # Same as circle but check that angle is in arc.
            return sorted(
                [
                    d / v.length
                    for d in ray_circle_intersection(ray, self.radius)
                    if is_in_mod_360(
                        ((d * v) + p).angle, angle0_deg, angle1_deg
                    )
                ]
            )

        return Trace(f)

    def get_envelope(self) -> Envelope:

        angle0_deg = self.angle0 * (180 / math.pi)
        angle1_deg = self.angle1 * (180 / math.pi)

        v1 = V2.polar(angle0_deg, self.radius)
        v2 = V2.polar(angle1_deg, self.radius)

        def wrapped(d: V2) -> SignedDistance:
            is_circle = abs(angle0_deg - angle1_deg) >= 360
            if is_circle or is_in_mod_360(d.angle, angle0_deg, angle1_deg):
                # Case 1: Point at arc
                return self.radius / d.length  # type: ignore
            else:
                # Case 2: Point outside of arc
                x: float = max(d.dot(v1), d.dot(v2))
                return x

        return Envelope(wrapped)

    def render(self, ctx: PyCairoContext) -> None:
        ctx.arc(0, 0, self.radius, self.angle0, self.angle1)

    def render_svg(self, dwg: Drawing) -> BaseElement:
        u = V2.polar(self.angle0 * (180 / math.pi), self.radius)
        v = V2.polar(self.angle1 * (180 / math.pi), self.radius)
        path = dwg.path(
            fill="none", style="vector-effect: non-scaling-stroke;"
        )

        angle0_deg = self.angle0 * (180 / math.pi)
        angle1_deg = self.angle1 * (180 / math.pi)

        large = 1 if (angle1_deg - angle0_deg) % 360 > 180 else 0
        path.push(
            f"M {u.x} {u.y} A {self.radius} {self.radius} 0 {large} 1 {v.x} {v.y}"
        )
        return path

    def render_tikz(self, pylatex: PyLatex, style: Style) -> PyLatexElement:
        start = 180 * (self.angle0 / math.pi)
        end = 180 * (self.angle1 / math.pi)
        return pylatex.TikZDraw(
            [
                pylatex.TikZCoordinate(
                    self.radius * math.cos(self.angle0),
                    self.radius * math.sin(self.angle0),
                ),
                "arc",
            ],
            options=pylatex.TikZOptions(
                radius=self.radius,
                **{"start angle": start, "end angle": end},
                **style.to_tikz(pylatex),
            ),
        )


@dataclass
class Text(Shape):
    """Text class."""

    text: str
    font_size: Optional[float]

    def get_bounding_box(self) -> BoundingBox:
        # Text doesn't have a bounding box since we can't accurately know
        # its size for all backends.
        eps = 1e-4
        self.bb = BoundingBox([origin, origin + P2(eps, eps)])
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
            style=f"""text-align:center; text-anchor:middle; dominant-baseline:middle;
                      font-family:sans-serif; font-weight: bold;
                      font-size:{self.font_size}px;
                      vector-effect: non-scaling-stroke;""",
        )

    def render_tikz(self, pylatex: PyLatex, style: Style) -> PyLatexElement:
        opts = {}
        opts["font"] = "\\small\\sffamily"
        opts["scale"] = str(
            3.5 * (1 if self.font_size is None else self.font_size)
        )
        styles = style.to_tikz(pylatex)
        if styles["fill"] is not None:
            opts["text"] = styles["fill"]
        return pylatex.TikZNode(
            text=self.text,
            # Scale parameters based on observations
            options=pylatex.TikZOptions(**opts),
        )


def from_pil(
    im: PIL.Image,
    alpha: float = 1.0,
) -> PyCairoSurface:
    import cairo

    format: cairo.Format = cairo.FORMAT_ARGB32
    if "A" not in im.getbands():
        im.putalpha(int(alpha * 256.0))
    arr = bytearray(im.tobytes("raw", "BGRa"))
    surface = cairo.ImageSurface.create_for_data(
        arr, format, im.width, im.height
    )
    return surface


@dataclass
class Image(Shape):
    """Image class."""

    local_path: str
    url_path: Optional[str]

    def __post_init__(self) -> None:
        if self.local_path.endswith("svg"):
            import cairosvg

            out = BytesIO()
            cairosvg.svg2png(url=self.local_path, write_to=out)
        else:
            out = open(self.local_path, "rb")  # type:ignore

        self.im = PIL.Image.open(out)
        self.height = self.im.height
        self.width = self.im.width

    def get_bounding_box(self) -> BoundingBox:
        left = origin.x - self.width / 2
        top = origin.y - self.height / 2
        tl = P2(left, top)
        br = P2(left + self.width, top + self.height)
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


@dataclass
class Spacer(Shape):
    """Spacer class."""

    width: float
    height: float

    def get_bounding_box(self) -> BoundingBox:
        left = origin.x - self.width / 2
        top = origin.y - self.height / 2
        tl = P2(left, top)
        br = P2(left + self.width, top + self.height)
        return BoundingBox([tl, br])

    def render_tikz(self, pylatex: PyLatex, style: Style) -> PyLatexElement:
        left = origin.x - self.width / 2
        top = origin.y - self.height / 2
        return pylatex.TikZPath(
            [
                pylatex.TikZCoordinate(left, top),
                "rectangle",
                pylatex.TikZCoordinate(left + self.width, top + self.height),
            ]
        )


class Raw(Rect):  # type: ignore
    """Shape class.

    A fake SVG node for importing latex.
    """

    def __init__(self, st: str):
        self.xml = ET.fromstring(st)

    def get_xml(self) -> ET.Element:
        return self.xml


@dataclass
class Latex(Shape):
    """Latex class."""

    text: str

    def __post_init__(self) -> None:
        # Need to install latextools for this to run.
        import latextools

        # Border ensures no cropping.
        latex_eq = latextools.render_snippet(
            f"{self.text}",
            commands=[latextools.cmd.all_math],
            config=latextools.DocumentConfig(
                "standalone", {"crop=true,border=0.1cm"}
            ),
        )
        self.eq = latex_eq.as_svg()
        self.width = self.eq.width
        self.height = self.eq.height
        self.content = self.eq.content
        # From latextools Ensures no clash between multiple math statements
        id_prefix = f"embed-{hash(self.content)}-"
        self.content = (
            self.content.replace('id="', f'id="{id_prefix}')
            .replace('="url(#', f'="url(#{id_prefix}')
            .replace('xlink:href="#', f'xlink:href="#{id_prefix}')
        )

    def get_bounding_box(self) -> BoundingBox:
        left = origin.x - self.width / 2
        top = origin.y - self.height / 2
        tl = P2(left, top)
        br = P2(left + self.width, top + self.height)
        return BoundingBox(tl, br).scale(0.05)

    def render(self, ctx: PyCairoContext) -> None:
        raise NotImplementedError

    def render_svg(self, dwg: Drawing) -> BaseElement:
        dx, dy = -self.width / 2, -self.height / 2
        g = dwg.g(transform=f"scale(0.05) translate({dx} {dy})")
        g.add(Raw(self.content))
        return g
