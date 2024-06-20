from typing import Iterable, List, Optional, Tuple, Union

import chalk.transform as tx
from chalk.envelope import Envelope
from chalk.monoid import associative_reduce
from chalk.shapes import Path, Spacer
from chalk.transform import Floating, Scalars, V2_t
from chalk.types import Diagram

# Functions mirroring Diagrams.Combinators and Diagrams.2d.Combinators


def with_envelope(self: Diagram, other: Diagram) -> Diagram:
    return self.compose(other.get_envelope())


def close_envelope(self: Diagram) -> Diagram:
    env = self.get_envelope()
    return self.compose(Envelope.from_bounding_box(env.to_bounding_box()))


# with_trace, phantom,


def strut(width: Floating, height: Floating) -> Diagram:
    from chalk.core import Primitive

    return Primitive.from_shape(Spacer(tx.ftos(width), tx.ftos(height)))


def pad(self: Diagram, extra: Floating) -> Diagram:
    """Scale outward directed padding for a diagram.

    Be careful using this if your diagram is not centered.

    Args:
        self (Diagram): Diagram object.
        extra (float): Amount of padding to add.

    Returns:
        Diagram: A diagram object.
    """
    envelope = self.get_envelope()

    # def f(d: V2_t) -> Scalars:
    #     assert envelope is not None
    #     return envelope(d) * extra

    # new_envelope = Envelope(f, envelope.is_empty)
    return self#.compose(new_envelope)


def frame(self: Diagram, extra: Floating) -> Diagram:
    """Add outward directed padding for a diagram.
    This padding is applied uniformly on all sides.

    Args:
        self (Diagram): Diagram object.
        extra (float): Amount of padding to add.

    Returns:
        Diagram: A diagram object.
    """
    envelope = self.get_envelope()

    def f(d: V2_t) -> Scalars:
        assert envelope is not None
        return envelope(d) + extra

    new_envelope = Envelope(f, envelope.is_empty)
    return self.compose(new_envelope)


# extrudeEnvelope, intrudeEnvelope


def atop(self: Diagram, other: Diagram) -> Diagram:
    # envelope1 = self.get_envelope()
    # envelope2 = other.get_envelope()
    # new_envelope = envelope1 + envelope2
    return self.compose(None, other)


# beneath


def above(self: Diagram, other: Diagram) -> Diagram:
    return beside(self, other, tx.X.unit_y)


# appends


def beside(self: Diagram, other: Diagram, direction: V2_t) -> Diagram:
    return atop(self, juxtapose(self, other, direction))


def place_at(
    diagrams: Iterable[Diagram], points: List[Tuple[float, float]]
) -> Diagram:
    return concat(d.translate(x, y) for d, (x, y) in zip(diagrams, points))


def place_on_path(diagrams: Iterable[Diagram], path: Path) -> Diagram:
    return concat(d.translate_by(p) for d, p in zip(diagrams, path.points()))

Cat = Union[Iterable[Diagram], Diagram]
def cat(
    diagram: Cat, v: V2_t, sep: Optional[Floating] = None
) -> Diagram:
    if isinstance(diagram, Diagram):
        axes = diagram.size()
        axis = len(axes) - 1
        assert diagram.size() != ()
        diagram = diagram._normalize()
        import jax
        from functools import partial
        # def fn(a: Diagram, b: Diagram) -> Diagram:
        #     @partial(jax.vmap)
        #     def merge(a, b):
        #         b.get_envelope()(-v)
        #         new = a.juxtapose(b, v)
        #         return new
        #     return merge(a, b)
        def call_scan(diagram):
            @jax.vmap
            def offset(diagram):
                env = diagram.get_envelope()
                right = env(v)
                left  = env(-v)
                return right, left
            right, left = offset(diagram)
            off = tx.X.np.roll(right, 1) + left
            off = off.at[0].set(0)
            off = tx.X.np.cumsum(off, axis=0)
            @jax.vmap
            def translate(off, diagram):
                return diagram.translate_by(v * off[..., None, None])
            return translate(off, diagram)
            #return jax.lax.associative_scan(fn, diagram, axis=0).compose_axis()
        for a in range(axis):
            call_scan = jax.vmap(call_scan, in_axes=a, out_axes=a)
        return call_scan(diagram).compose_axis()

    else:
        diagrams = iter(diagram)
        start = next(diagrams, None)
        sep_dia = hstrut(sep).rotate(tx.angle(v))
        if start is None:
            return empty()

        def fn(a: Diagram, b: Diagram) -> Diagram:
            return a.beside(sep_dia, v).beside(b, v)

        return fn(start, associative_reduce(fn, diagrams, empty()))


def concat(diagrams: Iterable[Diagram]) -> Diagram:
    """
    Concat diagrams atop of each other with atop.

    Args:
        diagrams (Iterable[Diagram]): Diagrams to concat.

    Returns:
        Diagram: New diagram

    """
    from chalk.core import BaseDiagram

    if isinstance(diagram, Diagram):
        size = diagram.size()
        assert size != ()
        return diagram.compose_axis()
    else:
        return BaseDiagram.concat(diagrams)  # type: ignore


def empty() -> Diagram:
    "Create an empty diagram"
    from chalk.core import BaseDiagram

    return BaseDiagram.empty()


# CompaseAligned.

# 2D


def hstrut(width: Optional[Floating]) -> Diagram:
    from chalk.core import Primitive

    if width is None:
        return empty()
    return Primitive.from_shape(Spacer(tx.ftos(width), tx.ftos(0)))


def vstrut(height: Optional[Floating]) -> Diagram:
    from chalk.core import Primitive

    if height is None:
        return empty()
    return Primitive.from_shape(Spacer(tx.ftos(0), tx.ftos(height)))


def hcat(
    diagrams: Iterable[Diagram], sep: Optional[Floating] = None
) -> Diagram:
    """
    Stack diagrams next to each other with `besides`.

    Args:
        diagrams (Iterable[Diagram]): Diagrams to stack.
        sep (Optional[float]): Padding between diagrams.

    Returns:
        Diagram: New diagram

    """
    return cat(diagrams, tx.X.unit_x, sep)


def vcat(
    diagrams: Iterable[Diagram], sep: Optional[Floating] = None) -> Diagram:
    """
    Stack diagrams above each other with `above`.

    Args:
        diagrams (Iterable[Diagram]): Diagrams to stack.
        sep (Optional[float]): Padding between diagrams.

    Returns:
        Diagrams

    """
    return cat(diagrams, tx.X.unit_y, sep)


# Extra


def above2(self: Diagram, other: Diagram) -> Diagram:
    """Given two diagrams ``a`` and ``b``, ``a.above2(b)``
    places ``a`` on top of ``b``. This moves ``a`` down to
    touch ``b``.

    💡 ``a.above2(b)`` is equivalent to ``a // b``.

    Args:
        self (Diagram): Diagram object.
        other (Diagram): Another diagram object.

    Returns:
        Diagram: A diagram object.
    """
    return beside(other, self, -tx.X.unit_y)


def juxtapose_snug(self: Diagram, other: Diagram, direction: V2_t) -> Diagram:
    trace1 = self.get_trace()
    trace2 = other.get_trace()
    d1, m1 = trace1.trace_v(tx.X.origin, direction)
    d2, m2 = trace2.trace_v(tx.X.origin, -direction)
    assert m1.all()
    assert m2.all()
    d = d1 - d2
    t = tx.translation(d)
    return other.apply_transform(t)


def beside_snug(self: Diagram, other: Diagram, direction: V2_t) -> Diagram:
    return atop(self, juxtapose_snug(self, other, direction))


def juxtapose(self: Diagram, other: Diagram, direction: V2_t) -> Diagram:
    """Given two diagrams ``a`` and ``b``, ``a.juxtapose(b, v)``
    places ``b`` to touch ``a`` along angle ve .

    Args:
        self (Diagram): Diagram object.
        other (Diagram): Another diagram object.
        direction (V2_T): (Normalized) vector angle to juxtapose

    Returns:
        Diagram: Repositioned ``b`` diagram
    """
    envelope1 = self.get_envelope()
    envelope2 = other.get_envelope()
    d = envelope1.envelope_v(direction) - envelope2.envelope_v(-direction)
    t = tx.translation(d)
    return other.apply_transform(t)


def at_center(self: Diagram, other: Diagram) -> Diagram:
    """Center two given diagrams.

    💡 `a.at_center(b)` means center of ``a`` is translated
    to the center of ``b``, and ``b`` sits on top of
    ``a`` along the axis out of the plane of the image.

    💡 In other words, ``b`` occludes ``a``.

    Args:
        self (Diagram): Diagram object.
        other (Diagram): Another diagram object.

    Returns:
        Diagram: A diagram object.
    """
    envelope1 = self.get_envelope()
    envelope2 = other.get_envelope()
    t = tx.translation(envelope1.center)
    new_envelope = envelope1 + (envelope2.apply_transform(t))
    return self.compose(new_envelope, other.apply_transform(t))
