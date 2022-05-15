import math

from functools import reduce
from typing import Iterable, List, Tuple, Optional

from chalk.core import Diagram, Empty, Primitive
from chalk.shape import (
    Arc,
    Circle,
    Rectangle,
    Path,
    Text,
    Image,
    Spacer,
    Latex,
)
from chalk.point import Point, Vector
from chalk.trail import Trail

ignore = [Trail, Vector]


def empty() -> Diagram:
    return Empty()


def make_path(
    coords: List[Tuple[float, float]], arrow: bool = False
) -> Diagram:
    points = [Point(x, y) for x, y in coords]
    return Primitive.from_shape(Path(points, arrow))


def circle(radius: float) -> Diagram:
    return Primitive.from_shape(Circle(radius))


def arc(radius: float, angle0: float, angle1: float) -> Diagram:
    return Primitive.from_shape(Arc(radius, angle0, angle1))


def arc_between(
    point1: Tuple[float, float], point2: Tuple[float, float], height: float
) -> Diagram:
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
        shape: Diagram = make_path([(0, 0), (d, 0)])
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

        shape = (
            Primitive.from_shape(Arc(r, -θ, θ)).rotate(φ).translate(d / 2, dy)
        )
    return shape.rotate(v.angle).translate(p.x, p.y)


def polygon(sides: int, radius: float, rotation: float = 0) -> Diagram:
    coords = []
    n = sides + 1
    for s in range(n):
        # Rotate to align with x axis.
        t = 2.0 * math.pi * s / sides + (math.pi / 2 * sides) + rotation
        coords.append((radius * math.cos(t), radius * math.sin(t)))
    return make_path(coords)


def place_at(
    diagrams: Iterable[Diagram], points: List[Tuple[float, float]]
) -> Diagram:
    return concat(d.translate(x, y) for d, (x, y) in zip(diagrams, points))


def hrule(length: float) -> Diagram:
    return make_path([(-length / 2, 0), (length / 2, 0)])


def vrule(length: float) -> Diagram:
    return make_path([(0, -length / 2), (0, length / 2)])


def regular_polygon(sides: int, side_length: float) -> Diagram:
    return polygon(sides, side_length / (2 * math.sin(math.pi / sides)))


def triangle(width: float) -> Diagram:
    return regular_polygon(3, width)


def rectangle(
    width: float, height: float, radius: Optional[float] = None
) -> Diagram:
    return Primitive.from_shape(Rectangle(width, height, radius))


def image(local_path: str, url_path: Optional[str]) -> Diagram:
    return Primitive.from_shape(Image(local_path, url_path))


def square(side: float) -> Diagram:
    return Primitive.from_shape(Rectangle(side, side))


def text(t: str, size: Optional[float]) -> Diagram:
    return Primitive.from_shape(Text(t, font_size=size))


def latex(t: str) -> Diagram:
    return Primitive.from_shape(Latex(t))


def atop(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    return diagram1.atop(diagram2)


def beside(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    return diagram1.beside(diagram2)


def above(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    return diagram1.above(diagram2)


def concat(diagrams: Iterable[Diagram]) -> Diagram:
    return reduce(atop, diagrams, empty())


def hstrut(width: Optional[float]) -> Diagram:
    if width is None:
        return empty()
    return Primitive.from_shape(Spacer(width, 0))


def hcat(diagrams: Iterable[Diagram], sep: Optional[float] = None) -> Diagram:
    diagrams = iter(diagrams)
    start = next(diagrams, None)
    if start is None:
        return empty()
    return reduce(lambda a, b: a | hstrut(sep) | b, diagrams, start)


def vstrut(height: Optional[float]) -> Diagram:
    if height is None:
        return empty()
    return Primitive.from_shape(Spacer(0, height))


def vcat(diagrams: Iterable[Diagram], sep: Optional[float] = None) -> Diagram:
    diagrams = iter(diagrams)
    start = next(diagrams, None)
    if start is None:
        return empty()
    return reduce(lambda a, b: a / vstrut(sep) / b, diagrams, start)


def connect(diagram: Diagram, name1: str, name2: str) -> Diagram:
    return connect_outer(diagram, name1, "C", name2, "C")


def connect_outer(
    diagram: Diagram, name1: str, c1: str, name2: str, c2: str
) -> Diagram:
    bb1 = diagram.get_subdiagram_bounding_box(name1)
    bb2 = diagram.get_subdiagram_bounding_box(name2)
    assert bb1 is not None, f"Name {name1} not found"
    assert bb2 is not None, f"Name {name2} not found"
    points = [bb1.cardinal(c1), bb2.cardinal(c2)]
    return Primitive.from_shape(Path(points))
