import math

from functools import reduce
from typing import Any, Iterable, List, Optional, Tuple

from diagrams.core import Diagram, Empty, Primitive
from diagrams.point import Point
from diagrams.shape import (
    Arc,
    Circle,
    LineTo,
    MoveTo,
    Path,
    Rectangle,
    RoundedRectangle,
    Shape,
    Text,
)


def empty() -> Diagram:
    return Empty()


def make_path(coords: List[Tuple[float, float]]) -> Diagram:
    if len(coords) < 2:
        return empty()
    else:
        points = [Point(x, y) for x, y in coords]
        start, *rest = points
        elements = [MoveTo(start)] + [LineTo(p) for p in rest]
        return Primitive.from_shape(Path(elements))


def arc_between(point1: Tuple[float, float], point2: Tuple[float, float], height: float) -> Diagram:
    # This implementaion is based on the original diagrams' `arcBetween` function:
    # https://hackage.haskell.org/package/diagrams-lib-1.4.5.1/docs/src/Diagrams.TwoD.Arc.html#arcBetween
    p = Point(*point1)
    q = Point(*point2)

    # determine the arc's length and its radius
    h = abs(height)
    v = q - p
    d = v.norm()
    θ = math.acos((d ** 2 - 4 * h ** 2) / (d ** 2 + 4 * h ** 2))
    r = d / (2 * math.sin(θ))

    if height > 0:
        φ = -math.pi / 2
        dy = r - h
    else:
        φ = +math.pi / 2
        dy = h - r

    return (
        Primitive.from_shape(Arc(r, -θ, θ))
        .rotate(φ)
        .translate(d / 2, dy)
        .rotate(v.angle())
        .translate(p.x, p.y)
    )


def circle(radius: float) -> Diagram:
    return Primitive.from_shape(Circle(radius))


def polygon(sides: int, radius: float) -> Diagram:
    coords = []
    n = sides + 1
    for s in range(n):
        t = 2.0 * math.pi * s / sides
        coords.append([radius * math.cos(t),
                       radius * math.sin(t)])
    return make_path(coords)


def regular_polygon(sides: int, side_length:float) -> Diagram:
    return polygon(sides, side_length / (2 * math.sin(math.pi / sides)))


def triangle(width: float) -> Diagram:
    return regular_polygon(3, width)


def rectangle(width: float, height: float) -> Diagram:
    return Primitive.from_shape(Rectangle(width, height))


def rounded_rectangle(width: float, height: float, radius: float) -> Diagram:
    return Primitive.from_shape(RoundedRectangle(width, height, radius))


def square(side: float) -> Diagram:
    return Primitive.from_shape(Rectangle(side, side))


def text(t: str, size: Optional[float]) -> Diagram:
    return Primitive.from_shape(Text(t, font_size=size))


def atop(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    return diagram1.atop(diagram2)


def beside(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    return diagram1.beside(diagram2)


def above(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    return diagram1.above(diagram2)


def concat(diagrams: Iterable[Diagram]) -> Diagram:
    return reduce(atop, diagrams, empty())


def hcat(diagrams: Iterable[Diagram]) -> Diagram:
    return reduce(beside, diagrams, empty())


def vcat(diagrams: Iterable[Diagram]) -> Diagram:
    return reduce(above, diagrams, empty())
