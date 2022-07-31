from typing import Optional, Tuple, Union

from chalk.core import Primitive
from chalk.path import Path
from chalk.shape import Arc, Circle, Rectangle
from chalk.transform import P2, to_radians
from chalk.types import Diagram
from chalk.arrow import unit_arc_between

# Functions mirroring Diagrams.2d.Shapes


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
      angle0 (float): Starting cutoff in degrees.
      angle1 (float): Finishing cutoff in degrees.

    Returns:
      Diagram

    """
    return Primitive.from_shape(
        Arc(radius, to_radians(angle0), to_radians(angle1))
    )


def polygon(sides: int, radius: float, rotation: float = 0) -> Diagram:
    """
    Draw a polygon.

    Args:
       sides (int): Number of sides.
       radius (float): Internal radius.
       rotation: (int): Rotation in degress

    Returns:
       Diagram
    """
    return Primitive.from_shape(
        Path.polygon(sides, radius, to_radians(rotation))
    )


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


def square(side: float) -> Diagram:
    return Primitive.from_shape(Rectangle(side, side))


def arc_between(
    point1: Union[P2, Tuple[float, float]],
    point2: Union[P2, Tuple[float, float]],
    height: float,
) -> Diagram:
    """Makes an arc starting at point1 and ending at point2, with the midpoint
    at a distance of abs(height) away from the straight line from point1 to
    point2. A positive value of height results in an arc to the left of the
    line from point1 to point2; a negative value yields one to the right.
    The implementaion is based on the the function arcBetween from Haskell's
    diagrams:
    https://hackage.haskell.org/package/diagrams-lib-1.4.5.1/docs/src/Diagrams.TwoD.Arc.html#arcBetween
    """
    if not isinstance(point1, P2):
        p = P2(*point1)
    else:
        p = point1
    if not isinstance(point2, P2):
        q = P2(*point2)
    else:
        q = point2

    v = q - p
    d = v.length
    shape, _ = unit_arc_between(d, height)
    return shape.rotate(-v.angle).translate_by(p)
