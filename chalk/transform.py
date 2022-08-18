import math
from typing import Any, Protocol, TypeVar

from planar import Affine as Affine
from planar import BoundingBox, Point, Polygon, Ray, Vec2, Vec2Array


def from_radians(θ: float) -> float:
    t = (θ / math.pi) * 180
    return t


def to_radians(θ: float) -> float:
    t = (θ / 180) * math.pi
    return t


def remove_translation(aff: Affine) -> Affine:
    a, b, c, d, e, f = aff[:6]
    return Affine(a, b, 0, d, e, 0)


def remove_linear(aff: Affine) -> Affine:
    a, b, c, d, e, f = aff[:6]
    return Affine(1, 0, c, 0, 1, f)


def transpose_translation(aff: Affine) -> Affine:
    a, b, c, d, e, f = aff[:6]
    return Affine(a, d, 0, b, e, 0)


def apply_affine(aff: Affine, x: Any) -> Any:
    return aff * x


TTrans = TypeVar("TTrans", bound="TransformableProtocol")


class TransformableProtocol(Protocol):
    def scale(self: TTrans, α: float) -> TTrans:
        ...

    def scale_x(self: TTrans, α: float) -> TTrans:
        ...

    def scale_y(self: TTrans, α: float) -> TTrans:
        ...

    def rotate(self: TTrans, θ: float) -> TTrans:
        ...

    def rotate_rad(self: TTrans, θ: float) -> TTrans:
        ...

    def rotate_by(self: TTrans, turns: float) -> TTrans:
        ...

    def reflect_x(self: TTrans) -> TTrans:
        ...

    def reflect_y(self: TTrans) -> TTrans:
        ...

    def shear_y(self: TTrans, λ: float) -> TTrans:
        ...

    def shear_x(self: TTrans, λ: float) -> TTrans:
        ...

    def translate(self: TTrans, dx: float, dy: float) -> TTrans:
        ...

    def translate_by(self: TTrans, vector) -> TTrans:  # type: ignore
        ...

    def _app(self, t: Affine) -> TTrans:
        ...


class Transformable(TransformableProtocol):
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
        "Rotate by θ degrees counterclockwise"
        return self._app(Affine.rotation(θ))

    def rotate_rad(self: TTrans, θ: float) -> TTrans:
        "Rotate by θ radians counterclockwise"
        return self._app(Affine.rotation(from_radians(θ)))

    def rotate_by(self: TTrans, turns: float) -> TTrans:
        "Rotate by fractions of a circle (turn)."
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

Vec2._app = lambda x, y: y * x  # type: ignore
Vec2.shear_x = Transformable.shear_x  # type: ignore
Vec2.shear_y = Transformable.shear_y  # type: ignore
Vec2.scale = Transformable.scale  # type: ignore
Vec2.scale_x = Transformable.scale_x  # type: ignore
Vec2.scale_y = Transformable.scale_y  # type: ignore
Vec2.rotate = Transformable.rotate  # type: ignore
Vec2.rotate_by = Transformable.rotate_by  # type: ignore
Vec2.reflect_x = Transformable.reflect_x  # type: ignore
Vec2.reflect_y = Transformable.reflect_y  # type: ignore
V2 = Vec2


Vec2.translate = Transformable.translate  # type: ignore
Vec2.translate_by = Transformable.translate_by  # type: ignore
P2 = Point

origin = P2(0, 0)
unit_x = V2(1, 0)
unit_y = V2(0, 1)


def apply_p2_affine(aff: Affine, x: Point) -> Point:
    y: Point = aff * x
    return y


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


Affine.__mul__ = affine  # type: ignore

# Explicit rexport

__all__ = ["BoundingBox", "Polygon", "Vec2Array", "Ray"]
Affine
