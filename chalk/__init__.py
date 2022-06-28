import math
from functools import reduce
from typing import Iterable, List, Optional, Tuple

try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata  # type: ignore

from planar import Affine, BoundingBox, Point, Vec2

from chalk.core import Diagram, Empty, Primitive, unit_x, unit_y
from chalk.envelope import Envelope
from chalk.shape import (
    Arc,
    Circle,
    Image,
    Latex,
    Path,
    Rectangle,
    Spacer,
    Text,
)
from chalk.trail import Trail

# Set library name the same as on PyPI
# must be the same as setup.py:setup(name=?)
__libname__: str = "chalk-diagrams"  # custom dunder attribute
__version__ = metadata.version(__libname__)


ignore = [Trail, Vec2]


def empty() -> Diagram:
    return Empty()


def make_path(
    coords: List[Tuple[float, float]], arrow: bool = False
) -> Diagram:
    return Primitive.from_shape(Path.from_list_of_tuples(coords, arrow))


def circle(radius: float) -> Diagram:
    """
    Draw a circle.

    Args:
       radius (float): Radius.

    Returns:
       Diagram

    """
    return Primitive.from_shape(Circle(radius))


def arc(radius: float, angle0: float, angle1: float) -> Diagram:
    """
    Draw an arc.

    Args:
      radius (float): Circle radius.
      angle0 (float): Starting cutoff in radians.
      angle1 (float): Finishing cutoff in radians.

    Returns:
      Diagram

    """
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
        θ = math.acos((d**2 - 4 * h**2) / (d**2 + 4 * h**2))
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
    """
    Draw a polygon.

    Args:
       sides (int): Number of sides.
       radius (float): Internal radius.
       rotation: (int): Rotation in radians

    Returns:
       Diagram
    """
    return Primitive.from_shape(Path.polygon(sides, radius, rotation))


def regular_polygon(sides: int, side_length: float) -> Diagram:
    return Primitive.from_shape(Path.regular_polygon(sides, side_length))


def hrule(length: float) -> Diagram:
    return Primitive.from_shape(Path.hrule(length))


def vrule(length: float) -> Diagram:
    return Primitive.from_shape(Path.vrule(length))


def triangle(width: float) -> Diagram:
    return regular_polygon(3, width)


def rectangle(
    width: float, height: float, radius: Optional[float] = None
) -> Diagram:
    """
    Draw a rectangle.

    Args:
        width (float): Width
        height (float): Height
        radius (Optional[float]): Radius for rounded corners.

    Returns:
        Diagrams
    """
    return Primitive.from_shape(Rectangle(width, height, radius))


def image(local_path: str, url_path: Optional[str]) -> Diagram:
    return Primitive.from_shape(Image(local_path, url_path))


def square(side: float) -> Diagram:
    return Primitive.from_shape(Rectangle(side, side))


def text(t: str, size: Optional[float]) -> Diagram:
    """
    Draw some text.

    Args:
       t (str): The text string.
       size (Optional[float]): Size of the text.

    Returns:
       Diagram

    """
    return Primitive.from_shape(Text(t, font_size=size))


def latex(t: str) -> Diagram:
    return Primitive.from_shape(Latex(t))


def atop(
    diagram1: Diagram, diagram2: Diagram, direction: Optional[Vec2]
) -> Diagram:
    """
    Places `diagram2` atop `diagram1`. This is done such that their
    origins align.

    💡 ``a.atop(b)`` is equivalent to ``a + b``.

    Args:
        diagram1 (Diagram): Base diagram object.
        diagram2 (Diagram): Diagram placed atop.
        direction (Optional[Vec2]): Placement direction.

    Returns:
        Diagram: New diagram object.
    """
    return diagram1.atop(diagram2, direction)


def beside(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    """
    Places `diagram2` beside `diagram1`.

    💡 ``a.beside(b)`` is equivalent to ``a | b``.

    This is done by translating `diagram2` rightward.
    The origin of `diagram1` remains.

    Args:
        diagram1 (Diagram): Left diagram object.
        diagram2 (Diagram): Right diagram object.

    Returns:
        Diagram: New diagram object.
    """
    return diagram1.beside(diagram2)


def place_at(
    diagrams: Iterable[Diagram], points: List[Tuple[float, float]]
) -> Diagram:
    return concat(d.translate(x, y) for d, (x, y) in zip(diagrams, points))


def place_on_path(diagrams: Iterable[Diagram], path: Path) -> Diagram:
    return concat(d.translate(p.x, p.y) for d, p in zip(diagrams, path.points))


def above(diagram1: Diagram, diagram2: Diagram) -> Diagram:
    """
    Places `diagram1` above `diagram2`.

    💡 ``a.above(b)`` is equivalent to ``a / b``.

    This is done by translating `diagram2` downward.
    The origin of `diagram1` remains.

    Args:
        diagram1 (Diagram): Top diagram object.
        diagram2 (Diagram): Bottom diagram object.

    Returns:
        Diagram: New diagram object.
    """
    return diagram1.above(diagram2)


def concat(
    diagrams: Iterable[Diagram], direction: Optional[Vec2] = None
) -> Diagram:
    """
    Concat diagrams atop of each other with atop.

    Args:
        diagrams (Iterable[Diagram]): Diagrams to concat.
        direction (Optional[Vec2]): Direction along envelope to place object.

    Returns:
        Diagram: New diagram

    """
    return reduce((lambda x, y: atop(x, y, direction)), diagrams, empty())


def strut(width: float, height: float) -> Diagram:
    return Primitive.from_shape(Spacer(width, height))


def hstrut(width: Optional[float]) -> Diagram:
    if width is None:
        return empty()
    return Primitive.from_shape(Spacer(width, 0))


def hcat(diagrams: Iterable[Diagram], sep: Optional[float] = None) -> Diagram:
    """
    Stack diagrams next to each other with `besides`.

    Args:
        diagrams (Iterable[Diagram]): Diagrams to stack.
        sep (Optional[float]): Padding between diagrams.

    Returns:
        Diagram: New diagram

    """
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
    """
    Stack diagrams above each other with `above`.

    Args:
        diagrams (Iterable[Diagram]): Diagrams to stack.
        sep (Optional[float]): Padding between diagrams.

    Returns:
        Diagrams

    """
    diagrams = iter(diagrams)
    start = next(diagrams, None)
    if start is None:
        return empty()
    return reduce(lambda a, b: a / vstrut(sep) / b, diagrams, start)


def cardinal(env: Envelope, dir: str) -> Point:
    """Returns the position of an edge or a corner of the Envelope
    based on a labeled direction (``dir``).

    Only really works for box envelopes.

    Args:
        env (Envelope): Envelope to check.
        dir (str): Direction of the edge or the corner.

    Choose `dir` from the following table.

    Click to expand:

        | `dir`  |   Directon   |  Type  |
        |:-------|:-------------|:------:|
        | ``N``  | North        | edge   |
        | ``S``  | South        | edge   |
        | ``W``  | West         | edge   |
        | ``E``  | East         | edge   |
        | ``NW`` | North West   | corner |
        | ``NE`` | North East   | corner |
        | ``SW`` | South West   | corner |
        | ``SE`` | South East   | corner |

    Returns:
        Point: A point object.

    """

    def min_point(env: Envelope) -> Point:
        return Point(-env(-unit_x), -env(-unit_y))

    def max_point(env: Envelope) -> Point:
        return Point(env(unit_x), env(unit_y))

    return {
        "N": unit_y * min_point(env),
        "S": unit_y * max_point(env),
        "W": unit_x * min_point(env),
        "E": unit_x * max_point(env),
        "NW": min_point(env),
        "NE": unit_y * min_point(env) + unit_x * max_point(env),
        "SW": unit_y * max_point(env) + unit_x * min_point(env),
        "SE": max_point(env),
        "C": env.center,
    }[dir]


def connect(diagram: Diagram, name1: str, name2: str) -> Diagram:
    return connect_outer(diagram, name1, "C", name2, "C")


def connect_outer(
    diagram: Diagram,
    name1: str,
    c1: str,
    name2: str,
    c2: str,
    arrow: bool = False,
) -> Diagram:
    bb1 = diagram.get_subdiagram_envelope(name1)
    bb2 = diagram.get_subdiagram_envelope(name2)
    assert bb1 is not None, f"Name {name1} not found"
    assert bb2 is not None, f"Name {name2} not found"
    points = [cardinal(bb1, c1), cardinal(bb2, c2)]
    return Primitive.from_shape(Path(points, arrow))
