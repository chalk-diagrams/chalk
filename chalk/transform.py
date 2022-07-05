import math
from typing import Any, TypeVar

from planar.py import Affine as Affine
from planar.py import BoundingBox, Point, Polygon, Ray, Vec2, Vec2Array


def to_cairo(affine: Affine) -> Any:
    import cairo

    def convert(a, b, c, d, e, f):  # type: ignore
        return cairo.Matrix(a, d, b, e, c, f)  # type: ignore

    return convert(*affine[:6])  # type: ignore


def to_svg(affine: Affine) -> str:
    def convert(
        a: float, b: float, c: float, d: float, e: float, f: float
    ) -> str:
        return f"matrix({a}, {d}, {b}, {e}, {c}, {f})"

    return convert(*affine[:6])


def to_tikz(affine: Affine) -> str:
    def convert(
        a: float, b: float, c: float, d: float, e: float, f: float
    ) -> str:
        return f"{{{a}, {d}, {b}, {e}, ({c}, {f})}}"

    return convert(*affine[:6])


def from_radians(θ: float) -> Affine:
    t = (θ / math.pi) * 180
    return t


def to_radians(θ: float) -> Affine:
    t = (θ / 180) * math.pi
    return t


def remove_translation(aff: Affine) -> Affine:
    a, b, c, d, e, f = aff[:6]
    return Affine(a, b, 0, d, e, 0)


def transpose_translation(aff: Affine) -> Affine:
    a, b, c, d, e, f = aff[:6]
    return Affine(a, d, 0, b, e, 0)


def apply_affine(aff: Affine, x: Any) -> Any:
    return aff * x


TTrans = TypeVar("TTrans", bound="Transformable")


class Transformable:
    """Transformable class."""

    def apply_transform(self, t: Affine) -> TTrans:
        pass

    def __rmul__(self, t: Affine) -> TTrans:
        return self._app(t)

    def _app(self, t: Affine) -> TTrans:
        return self.apply_transform(t)

    def scale(self: TTrans, α: float) -> TTrans:
        return self._app(Affine.scale(Vec2(α, α)))

    def scale_x(self: TTrans, α: float) -> TTrans:
        return self._app(Affine.scale(Vec2(α, 1)))

    def scale_y(self: TTrans, α: float) -> TTrans:
        return self._app(Affine.scale(Vec2(1, α)))

    def rotate(self: TTrans, θ: float) -> TTrans:
        "Rotate by θ degrees clockwise"
        return self._app(Affine.rotation(θ))

    def rotate_rad(self: TTrans, θ: float) -> TTrans:
        "Rotatte by θ radians clockwise"
        return self._app(Affine.rotation(from_radians(θ)))

    def rotate_by(self: TTrans, turns: float) -> TTrans:
        """Rotate by fractions of a circle (turn)."""
        θ = 2 * math.pi * turns
        return self._app(Affine.rotation(from_radians(θ)))

    def reflect_x(self: TTrans) -> TTrans:
        return self._app(Affine.scale(Vec2(-1, +1)))

    def reflect_y(self: TTrans) -> TTrans:
        return self._app(Affine.scale(Vec2(+1, -1)))

    def shear_y(self: TTrans, λ: float) -> TTrans:
        return self._app(Affine(1.0, 0.0, 0.0, λ, 1.0, 0.0))

    def shear_x(self: TTrans, λ: float) -> TTrans:
        return self._app(Affine(1.0, λ, 0.0, 0.0, 1.0, 0.0))

    def translate(self: TTrans, dx: float, dy: float) -> TTrans:
        return self._app(Affine.translation(Vec2(dx, dy)))

    def translate_by(self: TTrans, vector) -> TTrans:  # type: ignore
        return self._app(Affine.translation(vector))


# Below here are a collection of hacks to ensure that planar objects
# behave like the rest of the Chalk library. We do this by monkey
# patching in methods to Vec2 and by fixing a bug in the Affine
# transformation. This is not great, but necessary to keep the
# Object oriented api of Chalk.

Vec2._app = lambda x, y: y * x
Vec2.shear_x = Transformable.shear_x
Vec2.shear_y = Transformable.shear_y
Vec2.scale = Transformable.scale
Vec2.scale_x = Transformable.scale_x
Vec2.scale_y = Transformable.scale_y
Vec2.rotate = Transformable.rotate
Vec2.rotate_by = Transformable.rotate_by
Vec2.reflect_x = Transformable.reflect_x
Vec2.reflect_y = Transformable.reflect_y
V2 = Vec2


Vec2.translate = Transformable.translate
Vec2.translate_by = Transformable.translate_by
P2 = Point

origin = P2(0, 0)
unit_x = V2(1, 0)
unit_y = V2(0, 1)


def affine(affine: Affine, other: Any) -> Any:
    sa, sb, sc, sd, se, sf, _, _, _ = affine[:]
    if isinstance(other, Affine):
        oa, ob, oc, od, oe, of, _, _, _ = other
        return tuple.__new__(
            Affine,
            (
                sa * oa + sb * od,
                sa * ob + sb * oe,
                sa * oc + sb * of + sc,
                sd * oa + se * od,
                sd * ob + se * oe,
                sd * oc + se * of + sf,
                0.0,
                0.0,
                1.0,
            ),
        )

    elif hasattr(other, "from_points"):
        # Point/vector array
        points = getattr(other, "points", other)
        try:
            return other.from_points(
                Point(px * sa + py * sb + sc, px * sd + py * se + sf)
                for px, py in points
            )
        except TypeError:
            return NotImplemented
    else:
        try:
            vx, vy = other
        except Exception:
            return NotImplemented
        return Vec2(vx * sa + vy * sb + sc, vx * sd + vy * se + sf)


Affine.__mul__ = affine

# Explicit rexport

__all__ = ["BoundingBox", "Polygon", "Vec2Array", "Ray"]
Affine
