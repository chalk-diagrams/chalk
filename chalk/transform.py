import math
from dataclasses import dataclass
from typing import TypeVar, Any

from affine import Affine


@dataclass
class Transform:
    """Transform class."""

    def __call__(self) -> Affine:
        raise NotImplementedError

    def to_cairo(self) -> Any:
        import cairo
        def convert(a, b, c, d, e, f):
            return cairo.Matrix(a, d, b, e, c, f)
        return convert(*self()[:6])

    def to_svg(self) -> str:
        def convert(a, b, c, d, e, f):
            return f"matrix({a}, {d}, {b}, {e}, {c}, {f})"
        return convert(*self()[:6])

    def to_tikz(self) -> str:
        def convert(a, b, c, d, e, f):
            return f"{{{a}, {d}, {b}, {e}, ({c}, {f})}}"
        return convert(*self()[:6])


@dataclass
class Identity(Transform):
    """Identity class."""

    def __call__(self) -> Affine:
        return Affine.identity()


@dataclass
class Scale(Transform):
    """Scale class."""

    αx: float
    αy: float

    def __call__(self) -> Affine:
        return Affine.scale(self.αx, self.αy)


@dataclass
class Rotate(Transform):
    """Rotate class."""

    θ: float

    def __call__(self) -> Affine:
        t = (self.θ / math.pi) * 180
        return Affine.rotation(t)

    
@dataclass
class Translate(Transform):
    """Translate class."""

    dx: float
    dy: float

    def __call__(self) -> Affine:
        return Affine.translation(self.dx, self.dy)


@dataclass
class ShearX(Transform):
    """ShearX class."""

    λ: float

    def __call__(self) -> Affine:
        return Affine(1, self.λ, 0, 0, 1, 0)

@dataclass
class ShearY(Transform):
    """ShearY class."""

    λ: float

    def __call__(self) -> Affine:
        return Affine(1, 0, 0, self.λ, 1, 0)



@dataclass
class Compose(Transform):
    """Compose class."""

    t: Transform
    u: Transform

    def __call__(self) -> Affine:
        return self.t() * self.u()


TTrans = TypeVar("TTrans", bound="Transformable")


class Transformable:
    """Transformable class."""

    def apply_transform(self, t: Transform) -> TTrans:
        pass

    def scale(self: TTrans, α: float) -> TTrans:
        return self.apply_transform(Scale(α, α))

    def scale_x(self: TTrans, α: float) -> TTrans:
        return self.apply_transform(Scale(α, 1))

    def scale_y(self: TTrans, α: float) -> TTrans:
        return self.apply_transform(Scale(1, α))

    def rotate(self: TTrans, θ: float) -> TTrans:
        return self.apply_transform(Rotate(θ))

    def rotate_by(self: TTrans, turns: float) -> TTrans:
        """Rotate by fractions of a circle (turn)."""
        θ = 2 * math.pi * turns
        return self.apply_transform(Rotate(θ))

    def reflect_x(self: TTrans) -> TTrans:
        return self.apply_transform(Scale(-1, +1))

    def reflect_y(self: TTrans) -> TTrans:
        return self.apply_transform(Scale(+1, -1))

    def shear_x(self: TTrans, λ: float) -> TTrans:
        return self.apply_transform(ShearX(λ))

    def shear_y(self: TTrans, λ: float) -> TTrans:
        return self.apply_transform(ShearY(λ))

    def translate(self: TTrans, dx: float, dy: float) -> TTrans:
        return self.apply_transform(Translate(dx, dy))

    def translate_by(self: TTrans, vector) -> TTrans:  # type: ignore
        return self.apply_transform(Translate(vector.dx, vector.dy))
