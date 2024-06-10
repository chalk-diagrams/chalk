from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Optional

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
        return Envelope(lambda v: tx.X.np.array(0.0), is_empty=True)

    def __add__(self, other: Envelope) -> Envelope:
        if self.is_empty:
            return other
        if other.is_empty:
            return self
        return Envelope(
            lambda direction: tx.X.np.maximum(self(direction), other(direction))
        )

    all_dir = tx.X.np.concatenate(
        [tx.X.unit_x, -tx.X.unit_x, tx.X.unit_y, -tx.X.unit_y], axis=0
    )

    @property
    def center(self) -> P2_t:
        if self.is_empty:
            return tx.X.origin
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

    @staticmethod
    def general_transform(t: Affine, fn) -> Envelope: # type: ignore
        rt = tx.remove_translation(t)
        inv_t = tx.inv(rt)
        trans_t = tx.transpose_translation(rt)
        u: V2_t = -tx.get_translation(t)

        def wrapped(v: V2_t) -> tx.Scalars:
            # Linear
            v = v[:, None, :, :]

            vi = inv_t @ v
            inp = trans_t @ v
            v_prim = tx.norm(inp)
            inner = fn(v_prim)
            d = tx.dot(v_prim, vi)
            after_linear = inner / d

            # Translation
            diff = tx.dot((u / tx.dot(v, v)[..., None, None]), v)
            out = after_linear - diff
            return tx.X.np.max(out, axis=-1)

        return Envelope(wrapped)

    def apply_transform(self, t: Affine) -> Envelope:
        if self.is_empty:
            return self

        def apply(x): # type: ignore
            return self.f(x[..., 0, :, :])[..., None]

        return Envelope.general_transform(t, apply)

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
            return v / tx.length(d)

        return Envelope(wrapped)

    def to_bounding_box(self: Envelope) -> BoundingBox:
        d = self(Envelope.all_dir)
        return tx.BoundingBox(V2(-d[1], -d[3]), V2(d[0], d[2]))

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
        self, diagram: Primitive, t: Affine
    ) -> Envelope:
        new_transform = t @ diagram.transform
        return diagram.shape.get_envelope().apply_transform(new_transform)

    def visit_compose(
        self, diagram: Compose, t: Affine
    ) -> Envelope:
        return diagram.envelope.apply_transform(t)

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine
    ) -> Envelope:
        n = t @ diagram.transform
        return diagram.diagram.accept(self, n)


def get_envelope(self: Diagram, t: Optional[Affine] = None) -> Envelope:
    if t is None:
        t = tx.X.ident
    return self.accept(GetEnvelope(), t)
