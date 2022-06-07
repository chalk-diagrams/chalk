import math
from dataclasses import dataclass

from chalk import transform as tx


@dataclass
class Point(tx.Transformable):
    """Point class.

    This is a point in 2D and so has ``(x,y)``
    coordinates, and no ``z``-coordinate.
    """

    x: float
    y: float

    def apply_transform(self, t: tx.Transform):  # type: ignore
        """Applies a transformation to the point and
        returns the transformed point.

        Args:
            t (Transform): A transform object.

        Returns:
            Point: A point object.
        """
        new_x, new_y = t().transform_point(self.x, self.y)
        return Point(new_x, new_y)

    def __add__(self, other: "Vector") -> "Point":
        """Adds a vector to a point.

        Args:
            other (Vector): A vector object.

        Returns:
            Point: A point object.
        """
        return Point(self.x + other.dx, self.y + other.dy)

    def __sub__(self, other: "Point") -> "Vector":
        """Subtracts another point from this point and
        returns a vector.

        Args:
            other (Point): A point object.

        Returns:
            Vector: A vector object.
        """
        return Vector(self.x - other.x, self.y - other.y)


@dataclass
class Vector(tx.Transformable):
    """Vector class.

    This is a 2D vector.
    """

    dx: float
    dy: float

    @property
    def length(self) -> float:
        """Returns the length of the vector.

        Returns:
            float: Length of the vector.
        """
        return math.sqrt(self.dx**2 + self.dy**2)

    @property
    def angle(self) -> float:
        """Returns the angle of the vector.

        Returns:
            float: Angle of the vector (in radians).
        """
        return math.atan2(self.dy, self.dx)

    @classmethod
    def from_polar(cls, r: float, angle: float) -> "Vector":
        """Returns (constructs) a vector from polar-coordinate
        input ``(r, angle)``.

        Args:
            r (float): Length of the vector.
            angle (float): Angle of the vector (in radians).

        Returns:
            Vector: A vector object.
        """
        dx = r * math.cos(angle)
        dy = r * math.sin(angle)
        return cls(dx, dy)

    def apply_transform(self, t: tx.Transform):  # type:ignore
        """Applies a transformation on a vector
        and returns the transformed vector.

        Args:
            t (Transform): The transformation to apply.

        Returns:
            Vector: A vector object.
        """
        new_dx, new_dy = t().transform_point(self.dx, self.dy)
        return Vector(new_dx, new_dy)

    def rotate(self, by: float) -> "Vector":
        """Returns a rotated vector.

        Args:
            by (float): Angle of rotation (in radians)

        Returns:
            Vector: A vector object.
        """
        return Vector.from_polar(self.length, self.angle + by)

    def __mul__(self, α: float) -> "Vector":
        """Returns a scaled a vector.

        Args:
            α (float): Scaling factor.

        Returns:
            Vector: A vector object.
        """
        return Vector(α * self.dx, α * self.dy)

    __rmul__ = __mul__

    def __add__(self, other: "Vector") -> "Vector":
        """Adds another vector to this vector and
        returns the resulting vector.

        Args:
            other (Vector): Another vector.

        Returns:
            Vector: A vector object.
        """
        return Vector(self.dx + other.dx, self.dy + other.dy)

    def __sub__(self, other: "Vector") -> "Vector":
        """Subtracts another vector from this vector and
        returns the resulting vector.

        Args:
            other (Vector): Another vector.

        Returns:
            Vector: A vector object.
        """
        return Vector(self.dx - other.dx, self.dy - other.dy)

    def __neg__(self) -> "Vector":
        """Returns a negated vector.

        This flips the signs of the ``x`` and ``y`` components.

        Returns:
            Vector: A vector object.
        """
        return Vector(-self.dx, -self.dy)


ORIGIN = Point(0, 0)
