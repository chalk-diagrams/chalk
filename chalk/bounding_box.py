from dataclasses import dataclass

from chalk.point import ORIGIN, Point
from chalk.transform import Transform, Transformable


@dataclass
class BoundingBox(Transformable):
    """BoundingBox class."""

    tl: Point
    br: Point

    @classmethod
    def from_limits(
        cls, left: float, top: float, right: float, bottom: float
    ) -> "BoundingBox":
        """Returns a bounding box from limits
        (``left``, ``top``, ``right``, ``bottom``).

        Args:
            left (float): Position of left edge.
            top (float): Position of top edge.
            right (float): Position of right edge.
            bottom (float): Position of bottom edge.

        Returns:
            BoundingBox: A bounding box object.
        """
        tl = Point(left, top)
        br = Point(right, bottom)
        return cls(tl, br)

    @classmethod
    def empty(cls) -> "BoundingBox":
        """Returns an empty bounding box.

        The bounding box will have the following
        specifications:

        - top-left corner: `(0,0)`
        - bottom-right corner: `(0,0)`

        Returns:
            BoundingBox: A bounding box object.
        """
        return cls(ORIGIN, ORIGIN)

    @property
    def tr(self) -> Point:
        """Returns the position of the top-right corner
        of the bounding box.

        Returns:
            Point: A point object (``chalk.point.Point``).
        """
        return Point(self.right, self.top)

    @property
    def bl(self) -> Point:
        """Returns the position of the bottom-left corner
        of the bounding box.

        Returns:
            Point: A point object (``chalk.point.Point``).
        """
        return Point(self.left, self.bottom)

    def cardinal(self, dir: str) -> Point:
        """Returns the position of and edge or a corner of the bounding
        box based on a labeled direction (``dir``).

        Args:
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
            Point: A point object (``chalk.point.Point``).

        """
        return {
            "N": Point(self.left + self.width / 2, self.top),
            "S": Point(self.left + self.width / 2, self.bottom),
            "W": Point(self.left, self.top + self.height / 2),
            "E": Point(self.right, self.top + self.height / 2),
            "NW": Point(self.left, self.top),
            "NE": Point(self.right, self.top),
            "SW": Point(self.left, self.bottom),
            "SE": Point(self.right, self.bottom),
            "C": self.center,
        }[dir]

    @property
    def width(self) -> float:
        """Returns width of the bounding box.

        Returns:
            float: The width.
        """
        return self.br.x - self.tl.x

    @property
    def height(self) -> float:
        """Returns height of the bounding box.

        Returns:
            float: The height.
        """
        return self.br.y - self.tl.y

    @property
    def left(self) -> float:
        """Returns ``x`` position of the left-edge
        of the bounding box.

        Returns:
            float: The x-coordinate of the left edge.
        """
        return self.tl.x

    @property
    def top(self) -> float:
        """Returns ``y`` position of the top-edge
        of the bounding box.

        Returns:
            float: The y-coordinate of the top edge.
        """
        return self.tl.y

    @property
    def right(self) -> float:
        """Returns ``x`` position of the right-edge
        of the bounding box.

        Returns:
            float: The x-coordinate of the right edge.
        """
        return self.br.x

    @property
    def bottom(self) -> float:
        """Returns ``y`` position of the bottom-edge
        of the bounding box.

        Returns:
            float: The y-coordinate of the bottom edge.
        """
        return self.br.y

    @property
    def center(self) -> Point:
        """Returns position of the center
        of the bounding box.

        Returns:
            Point: A point object (``chalk.point.Point``).
        """
        x = (self.left + self.right) / 2
        y = (self.top + self.bottom) / 2
        return Point(x, y)

    def enclose(self, point: Point) -> "BoundingBox":
        """Return a bounding box that encloses a given point.

        Args:
            point (Point): A point object (``chalk.point.Point``).

        Returns:
            BoundingBox: A bounding box object.
        """
        return BoundingBox.from_limits(
            min(self.left, point.x),
            min(self.top, point.y),
            max(self.right, point.x),
            max(self.bottom, point.y),
        )

    def apply_transform(self, t: Transform) -> "BoundingBox":  # type: ignore
        """Applies a transformation to the bounding box.

        Args:
            t (Transform): A transform object (``chalk.transform.Transform``)

        Returns:
            BoundingBox: A bounding box object.
        """
        tl = self.tl.apply_transform(t)
        return (
            BoundingBox(tl, tl)
            .enclose(self.tr.apply_transform(t))
            .enclose(self.bl.apply_transform(t))
            .enclose(self.br.apply_transform(t))
        )

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Returns the union (merged bounding box) of this
        bounding box with another one.

        For two bounding boxes, ``a`` and ``b``, ``a.union(b)``
        will return another bounding box that minimally-contains
        both bounding boxes.

        Args:
            other (BoundingBox): Another bounding box object.

        Returns:
            BoundingBox: A bounding box object.
        """
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        return BoundingBox.from_limits(left, top, right, bottom)
