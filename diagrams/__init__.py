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
    """Makes an arc starting at point1 and ending at point2, with the midpoint
    at a distance of abs(height) away from the straight line from point1 to
    point2. A positive value of height results in an arc to the left of the
    line from point1 to point2; a negative value yields one to the right.

    The implementaion is based on the the function arcBetween from Haskell's
    diagrams:
    https://hackage.haskell.org/package/diagrams-lib-1.4.5.1/docs/src/Diagrams.TwoD.Arc.html#arcBetween

    """
    p = Point(*point1)
    q = Point(*point2)

    h = abs(height)
    v = q - p
    d = v.length

    if h < 1e-6:
        # Draw a line if the height is too small
        shape = make_path([(0, 0), (d, 0)])
    else:
        # Determine the arc's angle θ and its radius r
        θ = math.acos((d ** 2 - 4 * h ** 2) / (d ** 2 + 4 * h ** 2))
        r = d / (2 * math.sin(θ))

        if height > 0:
            # bend left
            φ = -math.pi / 2
            dy = r - h
        else:
            # bend right
            φ = +math.pi / 2
            dy = h - r

        shape = Primitive.from_shape(Arc(r, -θ, θ)).rotate(φ).translate(d / 2, dy)

    return shape.rotate(v.angle).translate(p.x, p.y)


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
