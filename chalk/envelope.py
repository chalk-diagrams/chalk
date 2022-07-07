from __future__ import annotations

from functools import reduce
from typing import Callable, Iterable

from chalk.transform import (
    P2,
    V2,
    Affine,
    BoundingBox,
    Polygon,
    Transformable,
    Vec2Array,
    apply_affine,
    origin,
    remove_translation,
    transpose_translation,
    unit_x,
    unit_y,
)

SignedDistance = float


class Envelope(Transformable):
    def __init__(
        self, f: Callable[[V2], SignedDistance], is_empty: bool = False
    ) -> None:
        self.f = f
        self.is_empty = is_empty

    def __call__(self, direction: V2) -> SignedDistance:
        assert not self.is_empty
        return self.f(direction)

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
        assert not self.is_empty
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

    @staticmethod
    def mappend(envelope1: Envelope, envelope2: Envelope) -> Envelope:
        return envelope1 + envelope2

    @staticmethod
    def empty() -> Envelope:
        return Envelope(lambda v: 0, is_empty=True)

    @staticmethod
    def concat(envelopes: Iterable[Envelope]) -> Envelope:
        return reduce(Envelope.mappend, envelopes, Envelope.empty())

    def apply_transform(self, t: Affine) -> Envelope:  # type: ignore
        if self.is_empty:
            return self
        _, _, c, _, _, f = t[:6]
        u = V2(c, f)
        t1 = transpose_translation(remove_translation(t))

        def wrapped(v: V2) -> SignedDistance:
            t1v = apply_affine(t1, v)
            t1v2 = t1v.scaled_to(1)
            return self(t1v2) * t1v.length + u.dot(v) / (v.dot(v))  # type: ignore

        return Envelope(wrapped)

    def envelope_v(self, v: V2) -> V2:
        if self.is_empty:
            return V2(0, 0)
        v = v.scaled_to(1)
        d = self(v)
        return d * v

    @staticmethod
    def from_bounding_box(box: BoundingBox) -> Envelope:
        def wrapped(d: V2) -> SignedDistance:
            v: float = apply_affine(
                Affine.rotation(d.angle), box
            ).bounding_box.max_point.x
            return v / d.length  # type: ignore

        return Envelope(wrapped)

    @staticmethod
    def from_circle(radius: float) -> Envelope:
        def wrapped(d: V2) -> SignedDistance:
            return radius / d.length  # type: ignore

        return Envelope(wrapped)

    @staticmethod
    def from_path(path: Vec2Array) -> Envelope:
        if len(path) > 2:
            hull = Polygon.convex_hull(path)

            def wrapped(d: V2) -> SignedDistance:
                v: float = apply_affine(
                    Affine.rotation(d.angle), hull
                ).bounding_box.max_point.x
                return v / d.length  # type: ignore

            return Envelope(wrapped)
        else:
            return Envelope.from_bounding_box(BoundingBox(path))

    def to_path(self, angle: int = 45) -> Vec2Array:
        "Draws an envelope by sampling every 10 degrees."
        pts = []
        for i in range(0, 361, angle):
            v = V2.polar(i)
            pts.append(self(v) * v)
        return Vec2Array(pts)

    def to_segments(self, angle: int = 45) -> Iterable[Vec2Array]:
        "Draws an envelope by sampling every 10 degrees."
        segments = []
        for i in range(0, 361, angle):
            v = V2.polar(i)
            segments.append(Vec2Array([origin, self(v) * v]))
        return segments
