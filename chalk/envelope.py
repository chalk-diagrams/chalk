from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Tuple

from chalk.transform import (
    P2,
    V2,
    Affine,
    BoundingBox,
    Transformable,
    apply_affine,
    origin,
    remove_translation,
    transpose_translation,
    unit_x,
    unit_y,
)
from chalk.monoid import Monoid
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import (
        ApplyTransform,
        Compose,
        Primitive,
    )
    from chalk.types import Diagram


SignedDistance = float
Ident = Affine.identity()


class Envelope(Transformable, Monoid):
    def __init__(
        self, f: Callable[[V2], SignedDistance], is_empty: bool = False
    ):
        self.f = f
        self.is_empty = is_empty

    def __call__(self, direction: V2) -> SignedDistance:
        assert not self.is_empty
        return self.f(direction)

    # Monoid
    @staticmethod
    def empty() -> Envelope:
        return Envelope(lambda v: 0, is_empty=True)

    def __add__(self, other: Envelope) -> Envelope:
        if self.is_empty:
            return other
        if other.is_empty:
            return self
        return Envelope(
            lambda direction: max(self(direction), other(direction))
        )

    @property
    def center(self) -> P2:
        if self.is_empty:
            return origin
        return P2(
            (-self(-unit_x) + self(unit_x)) / 2,
            (-self(-unit_y) + self(unit_y)) / 2,
        )

    @property
    def width(self) -> float:
        assert not self.is_empty
        return self(unit_x) + self(-unit_x)

    @property
    def height(self) -> float:
        assert not self.is_empty
        return self(unit_y) + self(-unit_y)


    def apply_transform(self, t: Affine) -> Envelope:
        if self.is_empty:
            return self
        rt = remove_translation(t)
        inv_t = ~rt
        trans_t = transpose_translation(rt)
        _, _, c, _, _, f = t[:6]
        u: V2 = -V2(c, f)

        def wrapped(v: V2) -> SignedDistance:
            # Linear
            vi = apply_affine(inv_t, v)
            v_prim = apply_affine(trans_t, v).normalized()
            inner: float = self(v_prim)
            d: float = v_prim.dot(vi)
            after_linear = inner / d

            # Translation
            diff: float = (u / (v.dot(v))).dot(v)
            return after_linear - diff

        return Envelope(wrapped)

    def envelope_v(self, v: V2) -> V2:
        if self.is_empty:
            return V2(0, 0)
        v = v.scaled_to(1)
        d: float = self(v)
        return v * d

    @staticmethod
    def from_bounding_box(box: BoundingBox) -> Envelope:
        def wrapped(d: V2) -> SignedDistance:
            v: float = apply_affine(
                Affine.rotation(d.angle), box
            ).bounding_box.max_point.x
            return v / d.length

        return Envelope(wrapped)

    @staticmethod
    def from_circle(radius: float) -> Envelope:
        def wrapped(d: V2) -> SignedDistance:
            return radius / d.length

        return Envelope(wrapped)

    def to_path(self, angle: int = 45) -> Iterable[P2]:
        "Draws an envelope by sampling every 10 degrees."
        pts = []
        for i in range(0, 361, angle):
            v = V2.polar(i)
            pts.append(self(v) * v)
        return pts

    def to_segments(self, angle: int = 45) -> Iterable[Tuple[P2, P2]]:
        "Draws an envelope by sampling every 10 degrees."
        segments = []
        for i in range(0, 361, angle):
            v = V2.polar(i)
            segments.append((origin, self(v) * v))
        return segments


class GetEnvelope(DiagramVisitor[Envelope, Affine]):
    A_type = Envelope
    def visit_primitive(
        self, diagram: Primitive, t: Affine = Ident
    ) -> Envelope:
        new_transform = t * diagram.transform
        return diagram.shape.get_envelope().apply_transform(new_transform)

    def visit_compose(self, diagram: Compose, t: Affine = Ident) -> Envelope:
        return diagram.envelope.apply_transform(t)

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine = Ident
    ) -> Envelope:
        n = t * diagram.transform
        return diagram.diagram.accept(self, n)


def get_envelope(self: Diagram, t: Affine = Ident) -> Envelope:
    return self.accept(GetEnvelope(), t)
