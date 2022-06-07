from typing import List

from chalk import transform as tx
from chalk.core import Primitive
from chalk.point import ORIGIN, Point, Vector
from chalk.shape import Path


class Trail(tx.Transformable):
    """Trail class.

    This is derived from a ``chalk.transform.Transformable`` class.

    [TODO]: Need more explanation on what this class is for (preferably
    with illustrations/figures).
    """

    def __init__(self, offsets: List[Vector]):
        self.offsets = offsets

    def __add__(self, other: "Trail") -> "Trail":
        """Adds another trail to this one and
        returns the resulting trail.

        Args:
            other (Trail): Another trail object.

        Returns:
            Trail: A trail object.
        """
        return Trail(self.offsets + other.offsets)

    @classmethod
    def from_path(cls, path: Path) -> "Trail":
        """Constructs and returns a trail from a given path.

        Args:
            path (Path): A path object.

        Returns:
            Trail: A trail object.
        """
        pts = path.points
        offsets = [t - s for s, t in zip(pts, pts[1:])]
        return cls(offsets)

    def to_path(self, origin: Point = ORIGIN) -> Path:
        """Converts a trail to a path, given a point (as a reference).

        Args:
            origin (Point, optional): A point object.
                                      Defaults to ORIGIN.

        Returns:
            Path: A path object.
        """
        points = [origin]
        for s in self.offsets:
            points.append(points[-1] + s)
        return Path(points)

    def stroke(self) -> Primitive:
        """Returns a primitive (shape) with strokes

        Returns:
            Primitive: A primitive object with strokes.
        """
        return Primitive.from_shape(self.to_path())

    def transform(self, t: tx.Transform) -> "Trail":
        """Applies a transform on the trail and
        returns the resulting trail.

        Args:
            t (Transform): A transform object.

        Returns:
            Trail: A trail object.
        """
        return Trail([p.apply_transform(t) for p in self.offsets])

    def apply_transform(self, t: tx.Transform) -> "Trail":  # type: ignore
        """Applies a given transform.

        This is the same as ``Trail.transform()`` method.

        Args:
            t (Transform): A transform object.

        Returns:
            Trail: A trail object.
        """
        return self.transform(t)


unit_x = Trail([Vector(1, 0)])
unit_y = Trail([Vector(0, 1)])
