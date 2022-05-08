import math

from functools import reduce
from typing import Iterable, List, Tuple

from diagrams.core import Diagram, Empty, Primitive
from diagrams.shape import Circle, Rectangle, Path, Text
from diagrams.point import Point


def empty() -> Diagram:
    return Empty()


def make_path(coords: List[Tuple[float, float]]) -> Diagram:
    points = [Point(x, y) for x, y in coords]
    return Primitive.from_shape(Path(points))


def circle(radius: float) -> Diagram:
    return Primitive.from_shape(Circle(radius))


def triangle(width: float) -> Diagram:
    coords = [
        (0, -width / math.sqrt(3)),
        (+width / 2.0, width / math.sqrt(3) * 0.5),
        (-width / 2.0, width / math.sqrt(3) * 0.5),
        (0, -width / math.sqrt(3)),
    ]
    return make_path(coords)


def rectangle(width: float, height: float) -> Diagram:
    return Primitive.from_shape(Rectangle(width, height))


def square(side: float) -> Diagram:
    return Primitive.from_shape(Rectangle(side, side))


def text(t: str) -> Diagram:
    return Primitive.from_shape(Text(t))


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
