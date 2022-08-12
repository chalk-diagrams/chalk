from typing import Optional, Tuple, Union

from chalk.shapes.arc import ArcSegment, arc_seg, arc_seg_angle  # noqa: F401
from chalk.shapes.arrowheads import ArrowHead, dart  # noqa: F401
from chalk.shapes.image import Image, from_pil, image  # noqa: F401
from chalk.shapes.latex import Latex, Raw, latex  # noqa: F401
from chalk.shapes.path import Path, make_path  # noqa: F401
from chalk.shapes.segment import Segment, seg  # noqa: F401
from chalk.shapes.shape import Shape, Spacer  # noqa: F401
from chalk.shapes.text import Text, text  # noqa: F401
from chalk.trail import SegmentLike, Trail  # noqa: F401
from chalk.transform import P2, V2
from chalk.types import Diagram

# Functions mirroring Diagrams.2d.Shapes


def circle(radius: float) -> Diagram:
    return Trail.circle().stroke().center_xy().scale(radius)


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
    return (
        ArcSegment(angle0, angle1 - angle0)
        .at(V2.polar(angle0, 1))
        .stroke()
        .scale(radius)
    )


# def polygon(sides: int, radius: float, rotation: float = 0) -> Diagram:
#     """
#     Draw a polygon.

#     Args:
#        sides (int): Number of sides.
#        radius (float): Internal radius.
#        rotation: (int): Rotation in degress

#     Returns:
#        Diagram
#     """
#     return Trail.polygon(sides, radius, to_radians(rotation)).stroke()


def regular_polygon(sides: int, side_length: float) -> Diagram:
    return Trail.regular_polygon(sides, side_length).stroke().center_xy()


def hrule(length: float) -> Diagram:
    return Trail.hrule(length).stroke().center_xy()


def vrule(length: float) -> Diagram:
    return Trail.vrule(length).stroke().center_xy()


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
    return Trail.rectangle(width, height).stroke().center_xy()


def square(side: float) -> Diagram:
    return rectangle(side, side)


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
    p = P2(*point1)
    q = P2(*point2)
    return arc_seg(q - p, height).at(p).stroke()


ignore = [Optional]
