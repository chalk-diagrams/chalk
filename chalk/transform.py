from dataclasses import dataclass

import cairo
import math


@dataclass
class Transform:
    def __call__(self) -> cairo.Matrix:
        raise NotImplementedError

    def to_svg(self) -> str:
        raise NotImplementedError


@dataclass
class Identity(Transform):
    def __call__(self) -> cairo.Matrix:
        return cairo.Matrix()

    def to_svg(self) -> str:
        return "scale(1)"


@dataclass
class Scale(Transform):
    αx: float
    αy: float

    def __call__(self) -> cairo.Matrix:
        matrix = cairo.Matrix()
        matrix.scale(self.αx, self.αy)
        return matrix

    def to_svg(self) -> str:
        return f"scale({self.αx} {self.αy})"


@dataclass
class Rotate(Transform):
    θ: float

    def __call__(self) -> cairo.Matrix:
        return cairo.Matrix.init_rotate(self.θ)

    def to_svg(self) -> str:
        t = (self.θ / math.pi) * 180
        return f"rotate({t})"


@dataclass
class Translate(Transform):
    dx: float
    dy: float

    def __call__(self) -> cairo.Matrix:
        matrix = cairo.Matrix()
        matrix.translate(self.dx, self.dy)
        return matrix

    def to_svg(self) -> str:
        return f"translate({self.dx} {self.dy})"


@dataclass
class ShearX(Transform):
    λ: float

    def __call__(self) -> cairo.Matrix:
        matrix = cairo.Matrix(1, 0, self.λ, 1, 0, 0)  # type: ignore
        return matrix

    def to_svg(self) -> str:
        return f"matrix(1 0 {self.λ} 1 0 0)"


@dataclass
class ShearY(Transform):
    λ: float

    def __call__(self) -> cairo.Matrix:
        matrix = cairo.Matrix(1, self.λ, 0, 1, 0, 0)  # type: ignore
        return matrix

    def to_svg(self) -> str:
        return f"matrix(1 {self.λ} 0 1 0 0)"


@dataclass
class Compose(Transform):
    t: Transform
    u: Transform

    def __call__(self) -> cairo.Matrix:
        # return self.t().multiply(self.u())
        return self.u().multiply(self.t())

    def to_svg(self) -> str:
        return self.t.to_svg() + " " + self.u.to_svg()


class Transformable:
    def apply_transform(self, t: Transform) -> "Transformable":
        pass

    def scale(self, α: float) -> "Transformable":
        return self.apply_transform(Scale(α, α))

    def scale_x(self, α: float) -> "Transformable":
        return self.apply_transform(Scale(α, 1))

    def scale_y(self, α: float) -> "Transformable":
        return self.apply_transform(Scale(1, α))

    def rotate(self, θ: float) -> "Transformable":
        return self.apply_transform(Rotate(θ))

    def rotate_by(self, turns: float) -> "Transformable":
        """Rotate by fractions of a circle (turn)."""
        θ = 2 * math.pi * turns
        return self.apply_transform(Rotate(θ))

    def reflect_x(self) -> "Transformable":
        return self.apply_transform(Scale(-1, +1))

    def reflect_y(self) -> "Transformable":
        return self.apply_transform(Scale(+1, -1))

    def shear_x(self, λ: float) -> "Transformable":
        return self.apply_transform(ShearX(λ))

    def shear_y(self, λ: float) -> "Transformable":
        return self.apply_transform(ShearY(λ))

    def translate(self, dx: float, dy: float) -> "Transformable":
        return self.apply_transform(Translate(dx, dy))

    def translate_by(self, vector) -> "Transformable":  # type: ignore
        return self.apply_transform(Translate(vector.dx, vector.dy))
