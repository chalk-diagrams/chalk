from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Tuple

from chalk.monoid import Monoid
from chalk.transform import (
    P2,
    V2,
    V2_t,
    P2_t,
    Affine,
    BoundingBox,
    Transformable,
    Scalar
)
import jax.numpy as np
import chalk.transform as tx
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Compose, Primitive
    from chalk.types import Diagram


SignedDistance = tx.Scalar


class Envelope(Transformable, Monoid):
    def __init__(
        self, f: Callable[[V2_t], SignedDistance], is_empty: bool = False
    ):
        self.f = f
        self.is_empty = is_empty

    def __call__(self, direction: V2_t) -> SignedDistance:
        assert not self.is_empty
        return self.f(direction)

    # Monoid
    @staticmethod
    def empty() -> Envelope:
        return Envelope(lambda v: np.array(0.0), is_empty=True)

    def __add__(self, other: Envelope) -> Envelope:
        if self.is_empty:
            return other
        if other.is_empty:
            return self
        return Envelope(
            lambda direction: np.maximum(self(direction), other(direction))
        )

    @property
    def center(self) -> P2_t:
        if self.is_empty:
            return tx.origin
        return P2(
            (-self(-tx.unit_x) + self(tx.unit_x)) / 2,
            (-self(-tx.unit_y) + self(tx.unit_y)) / 2,
        )

    @property
    def width(self) -> Scalar:
        assert not self.is_empty
        return (self(tx.unit_x) + self(-tx.unit_x)).reshape(())

    @property
    def height(self) -> Scalar:
        assert not self.is_empty
        return (self(tx.unit_y) + self(-tx.unit_y)).reshape(())

    def apply_transform(self, t: Affine) -> Envelope:
        if self.is_empty:
            return self
        rt = tx.remove_translation(t)
        inv_t = tx.inv(rt)
        trans_t = tx.transpose_translation(rt)
        u: V2_t = -tx.get_translation(t)
        def wrapped(v: V2_t) -> SignedDistance:
            # Linear
            vi = inv_t @ v
            v_prim = tx.norm(trans_t @ v)
            
            inner = self(v_prim)
            d = tx.dot(v_prim, vi)
            after_linear = inner / d

            # Translation
            diff = tx.dot((u / tx.dot(v, v)), v)
            return (after_linear - diff).reshape(())

        return Envelope(wrapped)

    def envelope_v(self, v: V2_t) -> V2_t:
        if self.is_empty:
            return V2(0, 0)
        v = tx.norm(v)
        d = self(v)
        return v * d

    @staticmethod
    def from_bounding_box(box: BoundingBox) -> Envelope:
        def wrapped(d: tx.V2_t) -> SignedDistance:
            v = box.rotate_rad(tx.rad(d)).br[0]
            return v / tx.length(d)
        
        return Envelope(wrapped)

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

    def to_segments(self, angle: int = 45) -> Iterable[Tuple[P2_t, P2_t]]:
        "Draws an envelope by sampling every 10 degrees."
        segments = []
        for i in range(0, 361, angle):
            v = tx.polar(i)
            segments.append((tx.origin, self(v) * v))
        return segments


class GetEnvelope(DiagramVisitor[Envelope, Affine]):
    A_type = Envelope

    def visit_primitive(
        self, diagram: Primitive, t: Affine = tx.ident
    ) -> Envelope:
        new_transform = t @ diagram.transform
        return diagram.shape.get_envelope().apply_transform(new_transform)

    def visit_compose(self, diagram: Compose, t: Affine = tx.ident) -> Envelope:
        return diagram.envelope.apply_transform(t)

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine = tx.ident
    ) -> Envelope:
        n = t @ diagram.transform
        return diagram.diagram.accept(self, n)


def get_envelope(self: Diagram, t: Affine = tx.ident) -> Envelope:
    return self.accept(GetEnvelope(), t)
