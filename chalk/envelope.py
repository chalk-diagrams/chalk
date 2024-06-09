from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.transform import (
    P2,
    V2,
    Affine,
    BoundingBox,
    P2_t,
    Scalars,
    Transformable,
    V2_t,
)
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Compose, Primitive
    from chalk.types import Diagram


# quantize = tx.np.linspace(-100, 100, 1000)
# mult = tx.np.array([1000, 1, 0])[None]


class Envelope(Transformable, Monoid):
    total_env = 0

    def __init__(self, f: Callable[[V2_t], Scalars], is_empty: bool = False):
        self.f = f
        self.is_empty = is_empty

    def __call__(self, direction: V2_t) -> Scalars:
        Envelope.total_env += 1
        assert not self.is_empty
        return self.f(direction)
        # v = (mult @ tx.np.digitize(direction, quantize)).reshape(1)[0]
        # if v not in self.cache:
        #     self.cache[v] = self.f(direction)
        # return self.cache[v]

    # Monoid
    @staticmethod
    def empty() -> Envelope:
        return Envelope(lambda v: tx.np.array(0.0), is_empty=True)

    def __add__(self, other: Envelope) -> Envelope:
        if self.is_empty:
            return other
        if other.is_empty:
            return self
        return Envelope(
            lambda direction: tx.np.maximum(self(direction), other(direction))
        )

    all_dir = tx.np.concatenate(
        [tx.unit_x, -tx.unit_x, tx.unit_y, -tx.unit_y], axis=0
    )

    @property
    def center(self) -> P2_t:
        if self.is_empty:
            return tx.origin
        # Get all the directions
        d = self(Envelope.all_dir)
        return P2(
            (-d[1] + d[0]) / 2,
            (-d[3] + d[2]) / 2,
        )

    @property
    def width(self) -> Scalars:
        assert not self.is_empty
        d = self(Envelope.all_dir[:2])
        return d.sum()

    @property
    def height(self) -> Scalars:
        assert not self.is_empty
        d = self(Envelope.all_dir[2:])
        return d.sum()

    def apply_transform(self, t: Affine) -> Envelope:
        if self.is_empty:
            return self
        rt = tx.remove_translation(t)
        inv_t = tx.inv(rt)
        trans_t = tx.transpose_translation(rt)
        u: V2_t = -tx.get_translation(t)

        def wrapped(v: V2_t) -> Scalars:
            # Linear
            vi = inv_t @ v
            v_prim = tx.norm(trans_t @ v)
            inner = self(v_prim)
            d = tx.dot(v_prim, vi)
            after_linear = inner / d

            # Translation
            diff = tx.dot((u / tx.dot(v, v)[..., None, None]), v)
            return after_linear - diff

        return Envelope(wrapped)

    def envelope_v(self, v: V2_t) -> V2_t:
        if self.is_empty:
            return V2(0, 0)
        v = tx.norm(v)
        d = self(v)
        return v * d

    @staticmethod
    def from_bounding_box(box: BoundingBox) -> Envelope:
        def wrapped(d: tx.V2_t) -> Scalars:
            v = box.rotate_rad(tx.rad(d)).br[:, 0, 0]
            r = v / tx.length(d)
            return r

        return Envelope(wrapped)

    def to_bounding_box(self: Envelope) -> BoundingBox:
        d = self(Envelope.all_dir)
        return tx.BoundingBox(V2(-d[1], -d[3]), V2(d[0], d[2]))

    # @staticmethod
    # def from_circle(radius: tx.Floating) -> Envelope:
    #     def wrapped(d: V2_t) -> SignedDistance:
    #         return radius / tx.length(d)

    #     return Envelope(wrapped)

    def to_path(self, angle: int = 45) -> Iterable[P2_t]:
        "Draws an envelope by sampling every 10 degrees."
        pts = []
        for i in range(0, 361, angle):
            v = tx.polar(i)
            pts.append(self(v) * v)
        return pts

    def to_segments(self, angle: int = 45) -> V2_t:
        "Draws an envelope by sampling every 10 degrees."
        v = tx.polar(tx.np.arange(0, 361, angle) * 1.0)
        return v * self(v)[:, None, None]


class GetEnvelope(DiagramVisitor[Envelope, Affine]):
    A_type = Envelope

    def visit_primitive(
        self, diagram: Primitive, t: Affine = tx.ident
    ) -> Envelope:
        new_transform = t @ diagram.transform
        return diagram.shape.get_envelope().apply_transform(new_transform)

    def visit_compose(
        self, diagram: Compose, t: Affine = tx.ident
    ) -> Envelope:
        return diagram.envelope.apply_transform(t)

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine = tx.ident
    ) -> Envelope:
        n = t @ diagram.transform
        return diagram.diagram.accept(self, n)


def get_envelope(self: Diagram, t: Affine = tx.ident) -> Envelope:
    return self.accept(GetEnvelope(), t)
