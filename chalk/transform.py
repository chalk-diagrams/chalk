import math
from typing import Any, TypeVar

from planar import Affine, Vec2


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


def remove_translation(aff: Affine) -> Affine:
    a, b, c, d, e, f = aff[:6]
    return Affine(a, d, 0, b, e, 0)


def _fix_affine(aff: Affine) -> Affine:
    a, b, c, d, e, f = aff[:6]
    return Affine(a, d, c, b, e, f)


def apply_affine(aff: Affine, x: Any) -> Any:
    return _fix_affine(aff) * x


TTrans = TypeVar("TTrans", bound="Transformable")


class Transformable:
    """Transformable class."""

    def apply_transform(self, t: Affine) -> TTrans:
        pass

    def __rmul__(self, t: Affine) -> TTrans:
        return self._app(t)

    def _app(self, t: Affine) -> TTrans:
        return self.apply_transform(_fix_affine(t))

    def scale(self: TTrans, α: float) -> TTrans:
        return self._app(Affine.scale(Vec2(α, α)))

    def scale_x(self: TTrans, α: float) -> TTrans:
        return self._app(Affine.scale(Vec2(α, 1)))

    def scale_y(self: TTrans, α: float) -> TTrans:
        return self._app(Affine.scale(Vec2(1, α)))

    def rotate(self: TTrans, θ: float) -> TTrans:
        return self._app(Affine.rotate(from_radians(θ)))

    def rotate_by(self: TTrans, turns: float) -> TTrans:
        """Rotate by fractions of a circle (turn)."""
        θ = 2 * math.pi * turns
        return self._app(Affine.rotate(from_radians(θ)))

    def reflect_x(self: TTrans) -> TTrans:
        return self._app(Affine.scale(Vec2(-1, +1)))

    def reflect_y(self: TTrans) -> TTrans:
        return self._app(Affine.scale(Vec2(+1, -1)))

    def shear_x(self: TTrans, λ: float) -> TTrans:
        return self._app(Affine.shear(from_radians(math.atan(λ)), 0))

    def shear_y(self: TTrans, λ: float) -> TTrans:
        return self._app(Affine.shear(0, from_radians(math.atan(λ))))

    def translate(self: TTrans, dx: float, dy: float) -> TTrans:
        return self._app(Affine.translation(Vec2(dx, dy)))

    def translate_by(self: TTrans, vector) -> TTrans:  # type: ignore
        return self._app(Affine.translation(vector))
