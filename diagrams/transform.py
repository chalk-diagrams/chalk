from dataclasses import dataclass

import cairo


@dataclass
class Transform:
    def __call__(self) -> cairo.Matrix:
        raise NotImplemented


@dataclass
class Identity(Transform):
    def __call__(self) -> cairo.Matrix:
        return cairo.Matrix()


@dataclass
class Scale(Transform):
    αx: float
    αy: float

    def __call__(self) -> cairo.Matrix:
        matrix = cairo.Matrix()
        matrix.scale(self.αx, self.αy)
        return matrix


@dataclass
class Rotate(Transform):
    θ: float

    def __call__(self) -> cairo.Matrix:
        return cairo.Matrix.init_rotate(self.θ)


@dataclass
class Translate(Transform):
    dx: float
    dy: float

    def __call__(self) -> cairo.Matrix:
        matrix = cairo.Matrix()
        matrix.translate(self.dx, self.dy)
        return matrix


@dataclass
class Compose(Transform):
    t: Transform
    u: Transform

    def __call__(self) -> cairo.Matrix:
        # return self.t().multiply(self.u())
        return self.u().multiply(self.t())
