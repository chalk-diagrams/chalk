import functools
import math

# TODO: This is a bit hacky, but not sure
# how to make things work with both numpy and jax
import os
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Tuple, Union

from jaxtyping import Bool, Float, Int
from typing_extensions import Self
import array_api_compat.numpy as onp
if TYPE_CHECKING:
    from chalk.namespace import _ArrayAPINameSpace
else:
    import numpy.array_api as _ArrayAPINameSpace


JAX_MODE = False

if not TYPE_CHECKING and not eval(os.environ.get("CHALK_JAX", "0")):
    ops = None
    import numpy as np
    Array = np.ndarray
    jnp = None
else:
    import jax.numpy as jnp
    from jax import config
    from jaxtyping import Array

    JAX_MODE = True
    config.update("jax_enable_x64", True)  # type: ignore
    config.update("jax_debug_nans", True)  # type: ignore


def set_jax_mode(v: bool) -> None:
    global JAX_MODE
    JAX_MODE = v


Affine = Float[Array, "*#B 3 3"]
Angles = Float[Array, "*#B 2"]
V2_t = Float[Array, "*#B 3 1"]
P2_t = Float[Array, "*#B 3 1"]
Scalars = Float[Array, "*#B"]
IntLike = Union[Int[Array, "*#B"], int]
Floating = Union[Scalars, float, int]
Mask = Bool[Array, "*#B"]
ColorVec = Float[Array, "#*B 3"]
Property = Float[Array, "#*B"]

TraceDistances = Tuple[Float[Array, "#B S"], Bool[Array, "#B S"]]




class _tx:

    @property
    def np(self) -> type[_ArrayAPINameSpace]:
        if JAX_MODE:

            import jax.numpy as jnp

            return jnp # type: ignore
        else:
            return onp # type: ignore

    @staticmethod
    def union(
        x: Tuple[Array, Array], y: Tuple[Array, Array]
    ) -> Tuple[Array, Array]:
        if isinstance(x, onp.ndarray):
            n1 = tx.np.concatenate([x[0], y[0]], axis=1)
            m = tx.np.concatenate([x[1], y[1]], axis=1)
            return n1, m
        else:
            n1 = tx.np.concatenate([x[0], y[0]], axis=1)
            m = tx.np.concatenate([x[1], y[1]], axis=1)
            return n1, m

    @staticmethod
    def union_axis(x: Tuple[Array, Array], axis: int) -> Tuple[Array, Array]:
        n = [
            tx.np.squeeze(x, axis=axis)
            for x in tx.np.split(x[0], x[0].shape[axis], axis=axis)
        ]
        m = [
            tx.np.squeeze(x, axis=axis)
            for x in tx.np.split(x[1], x[1].shape[axis], axis=axis)
        ]
        ret = functools.reduce(_tx.union, zip(n, m))
        return ret

    @staticmethod
    def index_update(arr, index, values):  # type:ignore
        """
        Update the array `arr` at the given `index` with `values`
        and return the updated array.
        Supports both NumPy and JAX arrays.
        """
        if isinstance(arr, onp.ndarray):
            # If the array is a NumPy array
            new_arr = arr.copy()
            new_arr[index] = values
            return new_arr
        else:
            # If the array is a JAX array
            return arr.at[index].set(values)

    @property
    def unit_x(self) -> V2_t:
        return self.np.asarray([1.0, 0.0, 0.0]).reshape((1, 3, 1))

    @property
    def unit_y(self) -> V2_t:
        return self.np.asarray([0.0, 1.0, 0.0]).reshape((1, 3, 1))

    @property
    def origin(self) -> V2_t:
        return self.np.asarray([0.0, 0.0, 1.0]).reshape((1, 3, 1))

    @property
    def ident(self) -> V2_t:
        return self.np.asarray(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1]]]
        ).reshape((1, 3, 3))


tx = _tx()


def ftos(f: Floating) -> Scalars:
    return tx.np.asarray(f, dtype=tx.np.double)#.reshape(-1)


def V2(x: Floating, y: Floating) -> V2_t:
    x, y, o = tx.np.broadcast_arrays(ftos(x), ftos(y), ftos(0.0))
    return tx.np.stack([x, y, o], axis=-1)[..., None]


def P2(x: Floating, y: Floating) -> P2_t:
    x, y, o = tx.np.broadcast_arrays(ftos(x), ftos(y), ftos(1.0))
    return tx.np.stack([x, y, o], axis=-1)[..., None]


def norm(v: V2_t) -> V2_t:
    return v / length(v)[..., None, None]


def length(v: P2_t) -> Scalars:
    return tx.np.sqrt(length2(v))


def length2(v: V2_t) -> Scalars:
    return (v * v)[..., :2, 0].sum(-1)


def angle(v: P2_t) -> Scalars:
    return from_rad * rad(v)


def rad(v: P2_t) -> Scalars:
    return tx.np.atan2(v[..., 1, 0], v[..., 0, 0])


def perpendicular(v: V2_t) -> V2_t:
    return tx.np.hstack([-v[..., 1, 0], v[..., 0, 0], v[..., 2, 0]])


def make_affine(
    a: Floating,
    b: Floating,
    c: Floating,
    d: Floating,
    e: Floating,
    f: Floating,
) -> Affine:
    vals = list([ftos(x) for x in [a, b, c, d, e, f, 0.0, 0.0, 1.0]])
    vals = tx.np.broadcast_arrays(*vals)
    x = tx.np.stack(vals, axis=-1)
    x = x.reshape(vals[0].shape + (3, 3))
    return x


def dot(v1: V2_t, v2: V2_t) -> Scalars:
    return (v1 * v2).sum(-1).sum(-1)


def cross(v1: V2_t, v2: V2_t) -> Scalars:
    return tx.np.cross(v1, v2)


def to_point(v: V2_t) -> P2_t:
    index = (Ellipsis, 2, 0)
    return tx.index_update(v, index, 1)  # type: ignore


def to_vec(p: P2_t) -> V2_t:
    index = (Ellipsis, 2, 0)
    return tx.index_update(p, index, 0)  # type: ignore


def polar(angle: Floating, length: Floating = 1.0) -> P2_t:
    rad = to_radians(angle)
    x, y = tx.np.cos(rad), tx.np.sin(rad)
    return V2(x * length, y * length)


def scale(vec: V2_t) -> Affine:
    x, y = dot(tx.unit_x, vec), dot(tx.unit_y, vec)
    return make_affine(x, 0.0, 0.0, 0.0, y, 0.0)


def translation(vec: V2_t) -> Affine:
    x, y = dot(tx.unit_x, vec), dot(tx.unit_y, vec)
    return make_affine(1.0, 0.0, x, 0.0, 1.0, y)


def get_translation(aff: Affine) -> V2_t:
    index = (Ellipsis, slice(0, 2), 0)
    base = tx.np.zeros((aff.shape[:-2]) + (3, 1))
    return tx.index_update(base, index, aff[..., :2, 2])  # type: ignore


def rotation(rad: Floating) -> Affine:
    rad = ftos(-rad)
    ca, sa = tx.np.cos(rad), tx.np.sin(rad)
    return make_affine(ca, -sa, 0.0, sa, ca, 0.0)


def inv(aff: Affine) -> Affine:
    det = tx.np.linalg.det(aff)
    # assert tx.np.all(tx.np.abs(det) > 1e-5), f"{det} {aff}"
    idet = 1.0 / det
    sa, sb, sc = aff[..., 0, 0], aff[..., 0, 1], aff[..., 0, 2]
    sd, se, sf = aff[..., 1, 0], aff[..., 1, 1], aff[..., 1, 2]
    ra = se * idet
    rb = -sb * idet
    rd = -sd * idet
    re = sa * idet
    return make_affine(ra, rb, -sc * ra - sf * rb, rd, re, -sc * rd - sf * re)


from_rad = 180 / math.pi


def from_radians(θ: Floating) -> Scalars:
    return ftos(θ) * from_rad


def to_radians(θ: Floating) -> Scalars:
    return (ftos(θ) / 180) * math.pi


def remove_translation(aff: Affine) -> Affine:
    index = (Ellipsis, slice(0, 1), 2)
    return tx.index_update(aff, index, 0)  # type: ignore


def remove_linear(aff: Affine) -> Affine:
    index = (Ellipsis, slice(0, 2), slice(0, 2))
    return tx.index_update(aff, index, tx.np.eye(2))  # type: ignore


def transpose_translation(aff: Affine) -> Affine:
    index = (Ellipsis, slice(0, 2), slice(0, 2))
    swap = aff[..., :2, :2].swapaxes(-1, -2)
    return tx.index_update(aff, index, swap)  # type: ignore


class Transformable:
    """Transformable class."""

    def apply_transform(self, t: Affine) -> Self:  # type: ignore[empty-body]
        pass

    def __rmatmul__(self, t: Affine) -> Self:
        return self._app(t)

    def __rmul__(self, t: Affine) -> Self:
        return self._app(t)

    def _app(self, t: Affine) -> Self:
        return self.apply_transform(t)

    def scale(self, α: Floating) -> Self:
        return self._app(scale(V2(α, α)))

    def scale_x(self, α: Floating) -> Self:
        return self._app(scale(V2(α, 1)))

    def scale_y(self, α: Floating) -> Self:
        return self._app(scale(V2(1, α)))

    def rotate(self, θ: Floating) -> Self:
        "Rotate by θ degrees counterclockwise"
        return self._app(rotation(to_radians(θ)))

    def rotate_rad(self, θ: Floating) -> Self:
        "Rotate by θ radians counterclockwise"
        return self._app(rotation((θ)))

    def rotate_by(self, turns: Floating) -> Self:
        "Rotate by fractions of a circle (turn)."
        θ = 2 * math.pi * turns
        return self._app(rotation((θ)))

    def reflect_x(self) -> Self:
        return self._app(scale(V2(-1, +1)))

    def reflect_y(self) -> Self:
        return self._app(scale(V2(+1, -1)))

    def shear_y(self, λ: Floating) -> Self:
        return self._app(make_affine(1.0, 0.0, 0.0, λ, 1.0, 0.0))

    def shear_x(self, λ: Floating) -> Self:
        return self._app(make_affine(1.0, λ, 0.0, 0.0, 1.0, 0.0))

    def translate(self, dx: Floating, dy: Floating) -> Self:
        return self._app(translation(V2(dx, dy)))

    def translate_by(self, vector) -> Self:  # type: ignore
        return self._app(translation(vector))


@dataclass
class Ray:
    pt: P2_t
    v: V2_t

    def point(self, len: Scalars) -> P2_t:
        return self.pt + len[..., None, None] * self.v


@dataclass
class BoundingBox(Transformable):
    tl: P2_t
    br: P2_t

    def apply_transform(self, t: Affine) -> Self:  # type: ignore
        tl = t @ self.tl
        br = t @ self.br
        tl2 = tx.np.minimum(tl, br)
        br2 = tx.np.maximum(tl, br)
        return BoundingBox(tl2, br2)  # type: ignore

    @property
    def width(self) -> Scalars:
        return (self.br - self.tl)[..., 0, 0]

    @property
    def height(self) -> Scalars:
        return (self.br - self.tl)[..., 1, 0]


def ray_circle_intersection(
    anchor: P2_t, direction: V2_t, circle_radius: Floating
) -> Tuple[Scalars, Mask]:
    """Given a ray and a circle centered at the origin, return the parameter t
    where the ray meets the circle, that is:

    ray t = circle θ

    The above equation is solved as follows:

    x + t v_x = r sin θ
    y + t v_y = r cos θ

    By squaring the equations and adding them we get

    (x + t v_x)² + (y + t v_y)² = r²,

    which is equivalent to the following equation:

    (v_x² + v_y²) t² + 2 (x v_x + y v_y) t + (x² + y² - r²) = 0

    This is a quadratic equation, whose solutions are well known.

    """
    a = length2(direction)
    b = 2 * dot(anchor, direction)
    c = length2(anchor) - circle_radius**2
    Δ = b**2 - 4 * a * c
    eps = 1e-10  # rounding error tolerance

    mid = (((-eps <= Δ) & (Δ < 0)))[..., None]
    mask = (Δ < -eps)[..., None] | (mid * tx.np.asarray([1, 0]))

    # Bump NaNs since they are going to me masked out.
    ret = tx.np.stack(
        [
            (-b - tx.np.sqrt(Δ + 1e9 * mask[..., 0])) / (2 * a),
            (
                -b
                + tx.np.sqrt(
                    tx.np.where(mid[..., 0], 0, Δ) + 1e9 * mask[..., 1]
                )
            )
            / (2 * a),
        ],
        -1,
    )

    ret = tx.np.where(mid, (-b / (2 * a))[..., None], ret)
    return ret.transpose(2, 0, 1), 1 - mask.transpose(2, 0, 1)

    # v = -b / (2 * a)
    # print(v.shape)
    # ret2 = tx.np.stack([v, tx.np.zeros(v.shape) + 10000], -1)
    # where2 = tx.np.where(((-eps <= Δ) & (Δ < eps))[..., None], ret2, ret)

    # return tx.np.where((Δ < -eps)[..., None], 10000, where2).transpose(2, 0, 1)
    # if
    #     # no intersection
    #     return []
    # elif -eps <= Δ < eps:
    #     # tangent
    #     return
    # else:
    #     # the ray intersects at two points
    #     return [
    #     ]


# Explicit rexport
X = tx
__all__ = ["X"]
