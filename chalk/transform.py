import math
from typing import Any, Union, Tuple, Optional, List

from typing_extensions import Self
from dataclasses import dataclass
import functools
from jaxtyping import Float, Bool, Array, Int
if False:
    ops = None
    import numpy as np 
else:
    from jax import ops
    import jax.numpy as np
    import jax

import jax
Affine = Float[Array, "#B 3 3"]
V2_t = Float[Array, "#B 3 1"]
P2_t = Float[Array, "#B 3 1"]
Scalars = Float[Array, "#B"]
Scalar = Float[Array, ""]
PtB = Float[Array, "... 3 1"]

ScalarsB = Float[Array, "..."]
ScalarB = Float[Array, "..."]

Floating = Union[Scalars, Scalar, float, int]
FloatingB = Union[ScalarsB, ScalarB, float, int]
Mask = Bool[Array, "#B"]
def ftos(f: FloatingB) -> ScalarsB:
    return np.array(f, dtype=np.double).reshape(-1)


def union(x, y):
    if ops is None:
        n1 = np.concatenate([x[0], y[0]], axis=1)
        m = np.concatenate([x[1], y[1]], axis=1)
        keep = np.any(m, axis=0)
        #return n1[:, keep], m[:, keep]
        return n1, m
    else:
        n1 = np.concatenate([x[0], y[0]], axis=1)
        m = np.concatenate([x[1], y[1]], axis=1)
        return n1, m


def union_axis(x, axis):
    n = [np.squeeze(x, axis=axis) for x in np.split(x[0], x[0].shape[axis], axis=axis)]
    m = [np.squeeze(x, axis=axis) for x in np.split(x[1], x[1].shape[axis], axis=axis)]
    ret =  functools.reduce(union, zip(n, m))
    return ret


def index_update(arr, index, values):
    """
    Update the array `arr` at the given `index` with `values` and return the updated array.
    Supports both NumPy and JAX arrays.
    """

    if ops is None:
        # If the array is a NumPy array
        new_arr = arr.copy()
        new_arr[index] = values
        return new_arr
    else:
        # If the array is a JAX array
        return arr.at[index].set(values)


def V2(x: Floating, y: Floating) -> V2_t:
    x, y, o = ftos(x), ftos(y), ftos(0.)
    res= np.stack([x, y, o.repeat(x.size)], 
                    axis=-1)[...,None]
    return res

def P2(x: Floating, y: Floating) -> P2_t:
    x, y, o = ftos(x), ftos(y), ftos(1.)
    return np.stack([x, y, o.repeat(x.size)], 
                    axis=-1)[..., None]

def norm(v: V2_t) -> V2_t:
    return v / length(v)[..., None, None]

def length(v: PtB) -> ScalarsB:
    return np.sqrt(length2(v))

# These are untyped variants that are needed for multibatch cases.
def length_(v):
    return length(v.reshape(-1, 3, 1)).reshape(*v.shape[:2])

def angle_(v):
    return angle(v.reshape(-1, 3, 1)).reshape(*v.shape[:2])

def norm_(v):
    return norm(v.reshape(-1, 3, 1)).reshape(*v.shape)

def polar_(v):
    return polar(v.reshape(-1)).reshape(*v.shape, 3, 1)

def length2_(v):
    return length2(v.reshape(-1, 3, 1)).reshape(*v.shape[:2])

def length2(v: PtB) -> ScalarsB:
    return (v * v)[..., :2, 0].sum(-1)

def angle(v: PtB) -> ScalarsB:
    return from_rad * rad(v)


def rad(v: PtB) -> ScalarsB:
    return np.arctan2(v[..., 1, 0], v[..., 0, 0])

def perpendicular(v: V2_t) -> V2_t:
    return np.hstack([-v[..., 1, 0], v[..., 0, 0], v[..., 2, 0]])


def make_affine(a: Floating, b:Floating, c:Floating, d:Floating, e:Floating, f:Floating) -> Affine:
    s = ftos(a).size
    vals = list([ftos(x) for x in [a, b, c, d, e, f, 0., 0., 1.]])
    for i in range(6, 9):
        vals[i] = vals[i].repeat(s)
    return np.stack(vals, axis=-1).reshape(-1, 3, 3)

ident = make_affine(1., 0., 0., 0., 1., 0.)

def dot(v1, v2):
    return (v1 * v2).sum(-1).sum(-1)

def cross(v1: V2_t, v2: V2_t) -> Scalars:
    return np.cross(v1, v2)

def to_point(v: Float[Array, "... 1"]) -> P2_t:
    index = (Ellipsis, 2, 0)
    return index_update(v, index, 1)

def polar(angle: FloatingB, 
          length: Floating = 1.0) -> PtB:
    rad = to_radians(angle)
    x, y = np.cos(rad), np.sin(rad)
    return V2(x * length, y * length)

def scale(vec: V2_t) -> Affine:
    index = (Ellipsis, np.arange(0, 2), np.arange(0, 2))
    return index_update(ident, index, vec[..., :2, 0])

def translation(vec: V2_t) -> Affine:
    index = (Ellipsis, slice(0, 2), 2)
    base = ident.repeat(vec.shape[0], axis=0)
    return index_update(base, index, vec[..., :2, 0])
    #return .at[..., :2, 2].set(vec[..., :2, 0])

def get_translation(aff: Affine) -> V2_t:
    index = (Ellipsis, slice(0, 2), 0)
    return index_update(np.zeros([aff.shape[0], 3, 1]), index, 
                        aff[..., :2, 2])

def rotation(rad: Floating) -> Affine:
    rad = ftos(rad)
    ca, sa = np.cos(rad), np.sin(rad)
    up = np.stack([ca, sa, -sa, ca], axis=-1).reshape(-1, 2, 2)
    base = ident.repeat(rad.shape[0], axis=0)
    index = (Ellipsis, slice(0, 2), slice(0, 2))
    return index_update(base, index, up)

def inv(aff: Affine) -> Affine:
    det = np.linalg.det(aff)
    #assert np.all(np.abs(det) > 1e-5), f"{det} {aff}"
    idet = 1.0 / det
    sa, sb, sc = aff[..., 0, 0], aff[..., 0, 1], aff[..., 0, 2]
    sd, se, sf = aff[..., 1, 0], aff[..., 1, 1], aff[..., 1, 2]
    ra = se * idet
    rb = -sb * idet
    rd = -sd * idet
    re = sa * idet
    return make_affine(ra, rb, -sc*ra - sf*rb,
                       rd, re, -sc*rd - sf*re)

from_rad = (180 / math.pi)
def from_radians(θ: FloatingB) -> ScalarsB:
    return ftos(θ)  * from_rad

def to_radians(θ: Floating) -> Scalars:
    return (ftos(θ) / 180) * math.pi

def remove_translation(aff: Affine) -> Affine:
    #aff.at[..., :1, 2].set(0)
    index = (Ellipsis, slice(0, 1), 2)
    return index_update(aff, index, 0)

def remove_linear(aff: Affine) -> Affine:
    # aff.at[..., :2, :2].set(np.eye(2))
    index = (Ellipsis, slice(0, 2), slice(0, 2))
    return index_update(aff, index, np.eye(2))

def transpose_translation(aff: Affine) -> Affine:
    index = (Ellipsis, slice(0, 2), slice(0, 2))
    return index_update(aff,  index, aff[..., :2, :2].transpose(0, 2, 1))

class Transformable:
    """Transformable class."""

    def apply_transform(self, t: Affine) -> Self:  # type: ignore[empty-body]
        pass

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
class BoundingBox(Transformable):
    tl: P2_t
    br: P2_t

    def apply_transform(self, t: Affine) -> Self:
        tl = t @ self.tl
        br = t @ self.br
        tl2 = np.minimum(tl, br)
        br2 = np.maximum(tl, br)
        return BoundingBox(tl2, br2)

    @property
    def width(self) -> Scalar:
        return (self.br - self.tl)[0, 0, 0]

    @property
    def height(self) -> Scalar:
        return (self.br - self.tl)[0, 1, 0]

origin = P2(0, 0)
unit_x = V2(1, 0)
unit_y = V2(0, 1)



def ray_ray_intersection(
    ray1: Tuple[P2_t, V2_t], ray2: Tuple[P2_t, V2_t]
) -> Optional[Tuple[float, float]]:
    """Given two rays

    ray₁ = λ t . p₁ + t v₁
    ray₂ = λ t . p₂ + t v₂

    the function returns the parameters t₁ and t₂ at which the two rays meet,
    that is:

    ray₁ t₁ = ray₂ t₂

    """
    u = ray2[0] - ray1[0]
    x1 = cross(ray1, ray2[1])
    x2 = cross(u, ray1[1])
    x3 = cross(u, ray2[1])
    if x1 == 0 and x2 != 0:
        # parallel
        return None
    elif x1 == 0 and x2 == 0:
        return 0.0, 0.0
    else:
        # intersecting or collinear
        return x3 / x1, x2 / x1


def ray_circle_intersection(anchor: Float[Array, "... 3 1"], 
                            direction: Float[Array, "... 3 1"], 
                            circle_radius) -> Tuple[Float[Array, "..."],
                                                    Int[Array, "..."]]:
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
    eps = 1e-12  # rounding error tolerance

    mid = (((-eps <= Δ) & (Δ < eps)))[..., None]
    mask = (Δ < -eps)[..., None] | (mid * np.array([1, 0]))

    # Bump NaNs since they are going to me masked out. 
    ret = np.stack([(-b - np.sqrt(Δ + 1e5 * mask[..., 0])) / (2 * a),
                    (-b + np.sqrt(Δ + 1e5 * mask[..., 1])) / (2 * a)], -1)

    ret = np.where(mid, (-b  / (2 * a))[..., None], ret)
    return ret.transpose(2, 0, 1), 1- mask.transpose(2, 0, 1)


    v = -b / (2 * a)
    print(v.shape)
    ret2 = np.stack([v,
                     np.zeros(v.shape) + 10000], -1)
    where2 = np.where( ((-eps <= Δ) & (Δ < eps))[..., None], 
        ret2, 
        ret
    )

    return np.where((Δ < -eps)[..., None],
                     10000, where2
             ).transpose(2, 0, 1)
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

__all__ = ["np"]