from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from chalk import transform as tx
from chalk.envelope import Envelope
from chalk.segment import Segment
from chalk.shape import Shape
from chalk.style import Style
from chalk.trace import Trace
from chalk.transform import P2
from chalk.types import (
    BaseElement,
    Diagram,
    Drawing,
    PyCairoContext,
    PyLatex,
    PyLatexElement,
    SegmentLike,
)


def make_path(segments):
    return Path.from_list_of_tuples(segments).stroke()


@dataclass
class Path(Shape, tx.Transformable):
    """Path class."""

    segments: List[SegmentLike]

    @classmethod
    def from_point(cls, point: P2) -> Path:
        return cls(Segment(point, point))

    @classmethod
    def from_points(cls, points: List[P2]) -> Path:
        return cls(
            list([Segment(pt1, pt2) for pt1, pt2 in zip(points, points[1:])])
        )

    @classmethod
    def from_pairs(cls, points: List[Tuple[P2, P2]]) -> Path:
        return cls(list([Segment(pt1, pt2) for pt1, pt2 in points]))

    @classmethod
    def from_list_of_tuples(cls, coords: List[Tuple[float, float]]) -> Path:
        points = list([P2(x, y) for x, y in coords])
        return cls.from_points(points)

    def points(self) -> List[P2]:
        points = []
        for i, seg in enumerate(self.segments):
            if i == 0:
                points.append(seg.p)
            points.append(seg.q)
        return points

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
        return Envelope.concat(
            segment.get_envelope() for segment in self.segments
        )

    def get_trace(self) -> Trace:
        return Trace.concat(segment.get_trace() for segment in self.segments)

    def stroke(self) -> Diagram:
        """Returns a primitive (shape) with strokes

        Returns:
            Diagram: A diagram.
        """
        from chalk.core import Primitive

        return Primitive.from_shape(self)

    def apply_transform(self, t: tx.Affine) -> Path:  # type: ignore
        return Path([segment.apply_transform(t) for segment in self.segments])

    def is_closed(self) -> bool:
        self.segments = list(self.segments)
        if self.segments:
            diff = self.segments[0].p - self.segments[-1].q
            if diff.length < 1e-3:
                return True
        return False

    def render(self, ctx: PyCairoContext, style: Style) -> None:

        for i, seg in enumerate(self.segments):
            if i == 0:
                p = seg.p
                ctx.move_to(p.x, p.y)
            seg.render_path(ctx)
        if self.is_closed():
            ctx.close_path()

    def render_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        line = dwg.path(
            style="vector-effect: non-scaling-stroke;",
        )
        for i, seg in enumerate(self.segments):
            if i == 0:
                p = seg.p
                line.push(f"M {p.x} {p.y}")
            line.push(seg.render_svg_path())
        if self.is_closed():
            line.push("Z")
        return line

    def render_tikz(self, pylatex: PyLatex, style: Style) -> PyLatexElement:
        pts = pylatex.TikZPathList()
        for i, seg in enumerate(self.segments):
            if i == 0:
                p = seg.p
                pts.append(pylatex.TikZCoordinate(p.x, p.y))
            seg.render_tikz_path(pts, pylatex)
            # pts.append("--")
            # pts.append(pylatex.TikZCoordinate(p.x, p.y))
        if self.is_closed():
            pts.append("--")
            pts._arg_list.append(pylatex.TikZUserPath("cycle"))
        return pylatex.TikZDraw(
            pts, options=pylatex.TikZOptions(**style.to_tikz(pylatex))
        )
