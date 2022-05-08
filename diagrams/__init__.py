import math

from functools import reduce
from typing import Iterable, List, Tuple, Optional

from diagrams.core import Diagram, Empty, Primitive
from diagrams.shape import Circle, Rectangle, RoundedRectangle, Path, Text
from diagrams.point import Point


def empty() -> Diagram:
    return Empty()


def make_path(coords: List[Tuple[float, float]]) -> Diagram:
    points = [Point(x, y) for x, y in coords]
    return Primitive.from_shape(Path(points))


def circle(radius: float) -> Diagram:
    return Primitive.from_shape(Circle(radius))


def polygon(sides: int, radius: float, rotation: float = 0) -> Diagram:
    coords = []
    n = sides + 1
    for s in range(n):
        # Rotate to align with x axis.
        t = 2.0 * math.pi * s / sides + (math.pi / 2 * sides) + rotation
        coords.append((radius * math.cos(t), radius * math.sin(t)))
    return make_path(coords)


def regular_polygon(sides: int, side_length: float) -> Diagram:
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
