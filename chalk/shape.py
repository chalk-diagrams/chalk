from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from chalk.envelope import Envelope
from chalk.segment import ray_circle_intersection
from chalk.style import Style
from chalk.trace import SignedDistance, Trace
from chalk.transform import P2, V2, BoundingBox, Ray, origin
from chalk.types import (
    BaseElement,
    Drawing,
    PyCairoContext,
    PyLatex,
    PyLatexElement,
)


@dataclass
class Shape:
    """Shape class."""

    def get_bounding_box(self) -> BoundingBox:
        raise NotImplementedError

    def get_envelope(self) -> Envelope:
        return Envelope.from_bounding_box(self.get_bounding_box())

    def get_trace(self) -> Trace:
        # default trace based on bounding box
        from chalk.path import Path

        box = self.get_bounding_box()
        return Path.rectangle(box.width, box.height).get_trace()

    def render(self, ctx: PyCairoContext, style: Style) -> None:
        pass

    def render_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        return dwg.g()

    def render_tikz(self, p: PyLatex, style: Style) -> PyLatexElement:
        return p.TikZScope()


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
        from chalk.path import Path

        return Path.rectangle(self.width, self.height).get_trace()

    def render(self, ctx: PyCairoContext, style: Style) -> None:
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

    def render_svg(self, dwg: Drawing, style: Style) -> BaseElement:
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

    def render(self, ctx: PyCairoContext, style: Style) -> None:
        ctx.arc(0, 0, self.radius, self.angle0, self.angle1)

    def render_svg(self, dwg: Drawing, style: Style) -> BaseElement:
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
