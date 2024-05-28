from typing import Optional, Tuple, Union

from chalk.shapes.arc import Segment, arc_seg, arc_seg_angle  # noqa: F401
from chalk.shapes.arrowheads import ArrowHead, dart  # noqa: F401
from chalk.shapes.image import Image, from_pil, image  # noqa: F401
from chalk.shapes.latex import Latex, latex  # noqa: F401
from chalk.shapes.path import Path, make_path  # noqa: F401
from chalk.shapes.shape import Shape, Spacer  # noqa: F401
from chalk.shapes.text import Text, text  # noqa: F401
from chalk.trail import Trail  # noqa: F401
from chalk.transform import P2_t, V2_t, P2, V2
import chalk.transform as tx
from chalk.types import Diagram

# Functions mirroring Diagrams.2d.Shapes


def hrule(length: float) -> Diagram:
    return Trail.hrule(length).stroke().center_xy()


def vrule(length: float) -> Diagram:
    return Trail.vrule(length).stroke().center_xy()


# def polygon(sides: int, radius: float, rotation: float = 0) -> Diagram:
#     """
#     Draw a polygon.

#     Args:
#        sides (int): Number of sides.
#        radius (float): Internal radius.
#        rotation: (int): Rotation in degrees

#     Returns:
#        Diagram
#     """
#     return Trail.polygon(sides, radius, to_radians(rotation)).stroke()


def regular_polygon(sides: int, side_length: float) -> Diagram:
    """Draws a regular polygon with given number of sides and given side
    length. The polygon is oriented with one edge parallel to the x-axis."""
    return Trail.regular_polygon(sides, side_length).centered().stroke()


def triangle(width: float) -> Diagram:
    """Draws an equilateral triangle with the side length specified by
    the ``width`` argument. The origin is the traingle's centroid."""
    return regular_polygon(3, width)


def rectangle(
    width: float, height: float, radius: Optional[float] = None
) -> Diagram:
    """
    Draws a rectangle.

    Args:
        width (float): Width
        height (float): Height
        radius (Optional[float]): Radius for rounded corners.

    Returns:
        Diagrams
    """
    if radius is None:
        return Trail.rectangle(width, height).stroke().center_xy()
    else:
        return (
            Trail.rounded_rectangle(width, height, radius).stroke().center_xy()
        )


def square(side: float) -> Diagram:
    """Draws a square with the specified side length. The origin is the
    center of the square."""
    return rectangle(side, side)


def circle(radius: tx.Floating) -> Diagram:
    "Draws a circle with the specified ``radius``."
    return Trail.circle().stroke().center_xy().scale(radius)


def arc(radius: tx.Floating, angle0: tx.Floating, angle1: tx.Floating) -> Diagram:
    """
    Draws an arc.

    Args:
      radius (float): Circle radius.
      angle0 (float): Starting cutoff in degrees.
      angle1 (float): Finishing cutoff in degrees.

    Returns:
      Diagram

    """
    return (
        arc_seg_angle(tx.ftos(angle0), tx.ftos(angle1 - angle0))
        .at(tx.polar(angle0))
        .stroke()
        .scale(radius)
    )


def arc_between(
    point1: Union[P2_t, Tuple[float, float]],
    point2: Union[P2_t, Tuple[float, float]],
    height: float,
) -> Diagram:
    """Makes an arc starting at point1 and ending at point2, with the midpoint
    at a distance of abs(height) away from the straight line from point1 to
    point2. A positive value of height results in an arc to the left of the
    line from point1 to point2; a negative value yields one to the right.
    The implementation is based on the the function arcBetween from Haskell's
    diagrams:
    https://hackage.haskell.org/package/diagrams-lib-1.4.5.1/docs/src/Diagrams.TwoD.Arc.html#arcBetween
    """
    p = tx.P2(*point1)
    q = tx.P2(*point2)
    return arc_seg(q - p, height).at(p).stroke()


ignore = [Optional]

__all__ = [
    "Segment",
    "Shape",
    "Spacer",
    "Text",
    "text",
    "Trail",
    "P2",
    "V2",
    "Diagram",
    "hrule",
    "vrule",
    "regular_polygon",
    "triangle",
    "rectangle",
    "square",
    "circle",
    "arc",
    "arc_between",
    "Latex",
    "Trail",
    "Path",
    "Image",
    "ArrowHead",
    "arc_seg",
    "dart",
    "from_pil",
    "make_path",
    "arc_seg_angle",
]
