from __future__ import annotations

import math
from dataclasses import dataclass

from chalk import transform as tx
from chalk.envelope import Envelope
from chalk.segment import Segment
from chalk.shape import Shape
from chalk.style import Style
from chalk.trace import Trace
from chalk.transform import P2, Vec2Array
from chalk.types import (
    BaseElement,
    Drawing,
    List,
    PyCairoContext,
    PyLatex,
    PyLatexElement,
    Tuple,
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

    def is_closed(self) -> bool:
        if self.points:
            diff = self.points[0] - self.points[-1]
            if diff.length < 1e-3:
                return True
        return False

    def render(self, ctx: PyCairoContext, style: Style) -> None:
        p, *rest = self.points
        ctx.move_to(p.x, p.y)
        for p in rest:
            ctx.line_to(p.x, p.y)
        if self.is_closed():
            ctx.close_path()

    def render_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        line = dwg.path(
            style="vector-effect: non-scaling-stroke;",
        )
        if self.points:
            p = self.points[0]
            line.push(f"M {p.x} {p.y}")
        for p in self.points:

            line.push(f"L {p.x} {p.y}")
        if self.is_closed():
            line.push("Z")
        return line

    def render_tikz(self, pylatex: PyLatex, style: Style) -> PyLatexElement:
        pts = pylatex.TikZPathList()
        for p in self.points:
            pts.append(pylatex.TikZCoordinate(p.x, p.y))
            pts.append("--")
        if self.is_closed():
            pts._arg_list.append(pylatex.TikZUserPath("cycle"))
        else:
            pts._arg_list = pts._arg_list[:-1]
        return pylatex.TikZDraw(
            pts, options=pylatex.TikZOptions(**style.to_tikz(pylatex))
        )
