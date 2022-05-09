import math

from functools import reduce
from typing import Iterable, List, Tuple, Optional

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

def polygon(sides: int, radius: float, rotation: float = 0) -> Diagram:
    coords = []
    n = sides + 1
    for s in range(n):
        # Rotate to align with x axis.
        t = 2.0 * math.pi * s / sides + (math.pi / 2 * sides) + rotation
        coords.append((radius * math.cos(t), radius * math.sin(t)))
    return make_path(coords)

def hrule(length: float):                                                                             
    return make_path([(-length / 2, 0), (length / 2, 0)])                                           
def vrule(length: float):                                                                              
    return make_path([(0, -length / 2), (0, length / 2)]) 

def regular_polygon(sides: int, side_length: float) -> Diagram:
    return polygon(sides, side_length / (2 * math.sin(math.pi / sides)))


def triangle(width: float) -> Diagram:
    return regular_polygon(3, width)

def rectangle(width: float, height: float, radius: Optional[float] = None) -> Diagram:
    return Primitive.from_shape(Rectangle(width, height, radius))


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


def connect(diagram: Diagram, name1: str, name2: str, curve: int = 0):
    return connect_outer(diagram, name1, "C", name2, "C", curve)


def connect_outer(
    diagram: Diagram, name1: str, c1: str, name2: str, c2: str):
    bb1 = diagram.get_subdiagram_bounding_box(name1)
    bb2 = diagram.get_subdiagram_bounding_box(name2)
    points = [bb1.cardinal(c1), bb2.cardinal(c2)]
    return Primitive.from_shape(Path(points))
