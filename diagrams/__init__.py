from functools import reduce
from typing import Any, List, Optional, Tuple

from diagrams.core import Diagram, Primitive
from diagrams.shape import Shape, Circle, Rectangle


def circle(size: float) -> Diagram:
    return Primitive.from_shape(Circle(size))


def square(size: float) -> Diagram:
    return Primitive.from_shape(Rectangle(size, size))


def beside(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    return diagram1.beside(diagram2)


def hcat(diagrams: List[Diagram]) -> Diagram:
    return reduce(beside, diagrams)
