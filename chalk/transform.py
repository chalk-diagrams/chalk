import math
from dataclasses import dataclass
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
    

def rotate_radians(θ: float) -> Affine:
    t = (θ / math.pi) * 180
    return Affine.rotation(t)

def shear_x(λ: float) -> Affine:
    return Affine(1, 0, 0, λ, 1, 0)

def shear_y(λ: float) -> Affine:
    return Affine(1, λ, 0, 0, 1, 0)

def remove_translation(aff: Affine) -> Affine:
    a, b, c, d, e, f = aff[:6]
    return Affine(a, d, 0, b, e, 0)


def _fix_affine(aff) -> Affine:
    a, b, c, d, e, f = aff[:6]
    return Affine(a, d, c, b, e, f)

def apply_affine(aff, x) -> Affine:
    return _fix_affine(aff) * x

TTrans = TypeVar("TTrans", bound="Transformable")


class Transformable:
    """Transformable class."""

    def apply_transform(self, t: Affine) -> TTrans:
        pass

    def __rmul__(self, t: Affine) -> TTrans:
        return self.apply_transform(_fix_affine(t))

    def scale(self: TTrans, α: float) -> TTrans:
        return Affine.scale(Vec2(α, α)) * self

    def scale_x(self: TTrans, α: float) -> TTrans:
        return Affine.scale(Vec2(α, 1)) * self

    def scale_y(self: TTrans, α: float) -> TTrans:
        return Affine.scale(Vec2(1, α)) * self

    def rotate(self: TTrans, θ: float) -> TTrans:
        return rotate_radians(θ) * self

    def rotate_by(self: TTrans, turns: float) -> TTrans:
        """Rotate by fractions of a circle (turn)."""
        θ = 2 * math.pi * turns
        return rotate_radians(θ) * self

    def reflect_x(self: TTrans) -> TTrans:
        return Affine.scale(Vec2(-1, +1)) * self

    def reflect_y(self: TTrans) -> TTrans:
        return Affine.scale(Vec2(+1, -1)) * self

    def shear_x(self: TTrans, λ: float) -> TTrans:
        return shear_x(λ) * self

    def shear_y(self: TTrans, λ: float) -> TTrans:
        return shear_y(λ) * self

    def translate(self: TTrans, dx: float, dy: float) -> TTrans:
        return Affine.translation(Vec2(dx, dy)) * self

    def translate_by(self: TTrans, vector) -> TTrans:  # type: ignore
        return Affine.translation(vector) * self
