import math

from functools import reduce
from typing import Any, List, Optional, Tuple

from diagrams.core import Diagram, Primitive
from diagrams.shape import Shape, Circle, Rectangle, Path
from diagrams.point import Point


def make_path(coords: List[Tuple[float, float]]) -> Diagram:
    points = [Point(x, y) for x, y in coords]
    return Primitive.from_shape(Path(points))


def circle(radius: float) -> Diagram:
    return Primitive.from_shape(Circle(radius))


def triangle(width: float) -> Diagram:
    coords = [
        (0, - width / math.sqrt(3)),
        (+ width / 2.0, width / math.sqrt(3) * 0.5),
        (- width / 2.0, width / math.sqrt(3) * 0.5),
        (0, - width / math.sqrt(3)),
    ]
    return make_path(coords)


def rectangle(width: float, height: float) -> Diagram:
    return Primitive.from_shape(Rectangle(width, height))


def square(side: float) -> Diagram:
    return Primitive.from_shape(Rectangle(side, side))


def atop(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    return diagram1.atop(diagram2)


def beside(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    return diagram1.beside(diagram2)


def concat(diagrams: List[Diagram]) -> Diagram:
    return reduce(atop, diagrams)


def hcat(diagrams: List[Diagram]) -> Diagram:
    return reduce(beside, diagrams)
