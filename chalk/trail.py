from __future__ import annotations

from dataclasses import dataclass

from chalk.path import Path
from chalk.transform import (
    P2,
    V2,
    Affine,
    Transformable,
    Vec2Array,
    apply_affine,
    remove_translation,
)
from chalk.types import Diagram


@dataclass
class Trail(Transformable):
    """Trail class.

    This is derived from a ``chalk.transform.Transformable`` class.

    [TODO]: Need more explanation on what this class is for (preferably
    with illustrations/figures).
    """

    offsets: Vec2Array

    @staticmethod
    def empty() -> Trail:
        return Trail(Vec2Array([]))

    def __add__(self, other: Trail) -> Trail:
        """Adds another trail to this one and
        returns the resulting trail.

        Args:
            other (Trail): Another trail object.

        Returns:
            Trail: A trail object.
        """
        new_vec = Vec2Array(self.offsets)
        new_vec.extend(other.offsets)
        return Trail(new_vec)

    @classmethod
    def from_path(cls, path: Path) -> Trail:
        """Constructs and returns a trail from a given path.

        Args:
            path (Path): A path object.

        Returns:
            Trail: A trail object.
        """
        pts = path.points
        offsets = [t - s for s, t in zip(pts, pts[1:])]
        return cls(offsets)

    def to_path(self, origin: P2 = P2(0, 0)) -> Path:
        """Converts a trail to a path, given a point (as a reference).

        Args:
            origin (P2, optional): A point object.
                                      Defaults to ORIGIN.

        Returns:
            Path: A path object.
        """
        points = [origin]
        for s in self.offsets:
            points.append(points[-1] + s)
        return Path(points)

    def stroke(self) -> Diagram:
        """Returns a primitive (shape) with strokes

        Returns:
            Diagram: A diagram.
        """
        from chalk.core import Primitive

        return Primitive.from_shape(self.to_path())

    def apply_transform(self, t: Affine) -> Trail:  # type: ignore
        """Applies a given transform.

        This is the same as ``Trail.transform()`` method.

        Args:
            t (Transform): A transform object.

        Returns:
            Trail: A trail object.
        """
        return Trail(apply_affine(remove_translation(t), self.offsets))


unit_x = Trail(Vec2Array([V2(1, 0)]))
unit_y = Trail(Vec2Array([V2(0, 1)]))
