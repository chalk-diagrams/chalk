from __future__ import annotations

from dataclasses import dataclass
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
    from chalk.core import ApplyTransform, Compose, Primitive, Empty
    from chalk.types import Diagram
    from chalk.types import Enveloped

# quantize = tx.np.linspace(-100, 100, 1000)
# mult = tx.np.array([1000, 1, 0])[None]

@dataclass
class EnvDistance(Monoid):
    d: Scalars

    def __add__(self, other: Self) -> Self:
        return EnvDistance(tx.X.np.maximum(self.d, other.d))

    @staticmethod
    def empty() -> EnvDistance:
        return EnvDistance(tx.X.np.asarray(-1e5))

@dataclass
class Envelope(Transformable, Monoid):
    diagram: Diagram
    affine: Affine

    def __call__(self, direction: V2_t) -> Scalars:
        def apply(x):  # type: ignore
            return self.diagram.accept(ApplyEnvelope(), x[..., 0, :, :]).d[..., None]

        return  Envelope.general_transform(self.affine, apply)(direction)

    # # Monoid
    @staticmethod
    def empty() -> Envelope:
        from chalk.core  import Empty
        return Envelope(Empty(), tx.X.ident)

    all_dir = tx.X.np.concatenate(
        [tx.X.unit_x, -tx.X.unit_x, tx.X.unit_y, -tx.X.unit_y], axis=0
    )

    @property
    def center(self) -> P2_t:
        # Get all the directions
        d = self(Envelope.all_dir)
        return P2(
            (-d[1] + d[0]) / 2,
            (-d[3] + d[2]) / 2,
        )

    @property
    def width(self) -> Scalars:
        #assert not self.is_empty
        d = self(Envelope.all_dir[:2])
        return d.sum()

    @property
    def height(self) -> Scalars:
        #assert not self.is_empty
        d = self(Envelope.all_dir[2:])
        return d.sum()

    @staticmethod
    def general_transform(t: Affine, fn) -> Envelope:  # type: ignore
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

        return wrapped

    def apply_transform(self, t: Affine) -> Envelope:
        if self.is_empty:
            return self

        def apply(x):  # type: ignore
            return self.f(x[..., 0, :, :])[..., None]

        return Envelope.general_transform(t, apply)

    def envelope_v(self, v: V2_t) -> V2_t:
        # if self.is_empty:
        #     return V2(0, 0)
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
        v = tx.polar(tx.X.np.arange(0, 361, angle) * 1.0)
        return v * self(v)[:, None, None]


class ApplyEnvelope(DiagramVisitor[EnvDistance, V2_t]):
    A_type = EnvDistance

    def visit_primitive(self, diagram: Primitive, t: V2_t) -> EnvDistance:
        def apply(x):  # type: ignore
            return diagram.shape.envelope(x[..., 0, :, :])[..., None]

        return EnvDistance(Envelope.general_transform(diagram.transform, apply)(t))


    def visit_apply_transform(self, diagram: ApplyTransform, t: V2_t) -> EnvDistance:
        def apply(x):  # type: ignore
            return diagram.diagram.accept(self, x[..., 0, :, :]).d[..., None]

        return EnvDistance(Envelope.general_transform(diagram.transform, apply)(t))

    def visit_empty(self, diagram: Empty, t: V2_t) -> EnvDistance:
        return EnvDistance(tx.X.np.asarray(0))

class GetEnvelope(DiagramVisitor[Envelope, Affine]):
    A_type = Envelope

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Envelope:

        new_transform = t @ diagram.transform
        if diagram.is_multi():
            # MultiPrimitive only work in jax mode.
            import jax

            def env(v: V2_t) -> Scalars:
                def inner(shape: Enveloped, transform: Affine) -> Scalars:
                    env = shape.get_envelope().apply_transform(transform)
                    return env(v)

                r = jax.vmap(inner)(diagram.shape, diagram.transform)
                return r.max(0)

            return Envelope(env)
        else:
            return Envelope(diagram, t) #.get_envelope().apply_transform(new_transform)

    def visit_compose(self, diagram: Compose, t: Affine) -> Envelope:
        if diagram.envelope is not None:
            return Envelope(diagram.envelope.diagram, t)
        return Envelope(diagram, t)

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine
    ) -> Envelope:
        n = t @ diagram.transform
        return diagram.diagram.accept(self, n)


def get_envelope(self: Diagram, t: Optional[Affine] = None) -> Envelope:
    if t is None:
        t = tx.X.ident
    return self.accept(GetEnvelope(), t)
