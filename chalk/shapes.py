from typing import Optional, Tuple, Union

from chalk.arrow import unit_arc_between
from chalk.path import Path, make_path
from chalk.arc import ArcSegment
from chalk.transform import P2, to_radians
from chalk.types import Diagram

# Functions mirroring Diagrams.2d.Shapes


def circle(radius: float) -> Diagram:
    """
    Draw a circle.

    Args:
       radius (float): Radius.

    Returns:
       Diagram

    """
    return (
        Path(
            [
                ArcSegment.arc_between(P2(-1, 0), P2(1, 0), 1),
                ArcSegment.arc_between(P2(1, 0), P2(-1, 0), 1),
            ]
        )
        .scale(radius)
        .stroke()
    )


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
    return Path(
        [ArcSegment(angle0, angle1 - angle0).rotate(angle0).scale(radius)]
    ).stroke()


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
    return Path.polygon(sides, radius, to_radians(rotation)).stroke()


def regular_polygon(sides: int, side_length: float) -> Diagram:
    return Path.regular_polygon(sides, side_length).stroke()


def hrule(length: float) -> Diagram:
    return Path.hrule(length).stroke()


def vrule(length: float) -> Diagram:
    return Path.vrule(length).stroke()


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
    return Path.rectangle(width, height).stroke()


def square(side: float) -> Diagram:
    return Path.rectangle(side, side).stroke()


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
    return Path(
        [
            ArcSegment.arc_between(p, q, height),
        ]
    ).stroke()
