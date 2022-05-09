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
class Compose(Transform):
    t: Transform
    u: Transform

    def __call__(self) -> cairo.Matrix:
        # return self.t().multiply(self.u())
        return self.u().multiply(self.t())

    def to_svg(self) -> str:
        return self.t.to_svg() + " " + self.u.to_svg()
