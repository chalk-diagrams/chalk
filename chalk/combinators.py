from functools import reduce
from typing import Iterable, List, Optional, Tuple

from chalk.envelope import Envelope
from chalk.shapes import Path, Spacer
from chalk.transform import V2, Affine, origin, unit_x, unit_y
from chalk.types import Diagram

# Functions mirroring Diagrams.Combinators and Diagrams.2d.Combinators


def with_envelope(self: Diagram, other: Diagram) -> Diagram:
    return self.compose(other.get_envelope())


# with_trace, phantom,


def strut(width: float, height: float) -> Diagram:
    from chalk.core import Primitive

    return Primitive.from_shape(Spacer(width, height))


def pad(self: Diagram, extra: float) -> Diagram:
    """Scale outward directed padding for a diagram.

    Be careful using this if your diagram is not centered.

    Args:
        self (Diagram): Diagram object.
        extra (float): Amount of padding to add.

    Returns:
        Diagram: A diagram object.
    """
    envelope = self.get_envelope()

    def f(d: V2) -> float:
        assert envelope is not None
        return envelope(d) * extra

    new_envelope = Envelope(f, envelope.is_empty)
    return self.compose(new_envelope)


def frame(self: Diagram, extra: float) -> Diagram:
    """Add outward directed padding for a diagram.
    This padding is applied uniformly on all sides.

    Args:
        self (Diagram): Diagram object.
        extra (float): Amount of padding to add.

    Returns:
        Diagram: A diagram object.
    """
    envelope = self.get_envelope()

    def f(d: V2) -> float:
        assert envelope is not None
        return envelope(d) + extra

    new_envelope = Envelope(f, envelope.is_empty)
    return self.compose(new_envelope)


# extrudeEnvelope, intrudeEnvelope


def atop(self: Diagram, other: Diagram) -> Diagram:
    envelope1 = self.get_envelope()
    envelope2 = other.get_envelope()
    new_envelope = envelope1 + envelope2
    return self.compose(new_envelope, other)


# beneath


def above(self: Diagram, other: Diagram) -> Diagram:
    return beside(self, other, unit_y)


# appends


def beside(self: Diagram, other: Diagram, direction: V2) -> Diagram:
    return atop(self, juxtapose(self, other, direction))


def place_at(
    diagrams: Iterable[Diagram], points: List[Tuple[float, float]]
) -> Diagram:
    return concat(d.translate(x, y) for d, (x, y) in zip(diagrams, points))


def place_on_path(diagrams: Iterable[Diagram], path: Path) -> Diagram:
    return concat(
        d.translate(p.x, p.y) for d, p in zip(diagrams, path.points())
    )


# position, atPoints


def cat(
    diagrams: Iterable[Diagram], v: V2, sep: Optional[float] = None
) -> Diagram:
    from chalk.core import empty

    diagrams = iter(diagrams)
    start = next(diagrams, None)
    sep_dia = hstrut(sep).rotate(v.angle)
    if start is None:
        return empty()
    return reduce(
        lambda a, b: a.beside(sep_dia, v).beside(b, v), diagrams, start
    )


def concat(diagrams: Iterable[Diagram]) -> Diagram:
    """
    Concat diagrams atop of each other with atop.

    Args:
        diagrams (Iterable[Diagram]): Diagrams to concat.

    Returns:
        Diagram: New diagram

    """
    from chalk.core import empty

    return reduce(atop, diagrams, empty())


# CompaseAligned.

# 2D


def hstrut(width: Optional[float]) -> Diagram:
    from chalk.core import Primitive, empty

    if width is None:
        return empty()
    return Primitive.from_shape(Spacer(width, 0))


def vstrut(height: Optional[float]) -> Diagram:
    from chalk.core import Primitive, empty

    if height is None:
        return empty()
    return Primitive.from_shape(Spacer(0, height))


def hcat(diagrams: Iterable[Diagram], sep: Optional[float] = None) -> Diagram:
    """
    Stack diagrams next to each other with `besides`.

    Args:
        diagrams (Iterable[Diagram]): Diagrams to stack.
        sep (Optional[float]): Padding between diagrams.

    Returns:
        Diagram: New diagram

    """
    return cat(diagrams, unit_x, sep)


def vcat(diagrams: Iterable[Diagram], sep: Optional[float] = None) -> Diagram:
    """
    Stack diagrams above each other with `above`.

    Args:
        diagrams (Iterable[Diagram]): Diagrams to stack.
        sep (Optional[float]): Padding between diagrams.

    Returns:
        Diagrams

    """
    return cat(diagrams, unit_y, sep)


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
    return beside(other, self, -unit_y)


def juxtapose_snug(self: Diagram, other: Diagram, direction: V2) -> Diagram:
    trace1 = self.get_trace()
    trace2 = other.get_trace()
    d1 = trace1.trace_v(origin, direction)
    d2 = trace2.trace_v(origin, -direction)
    assert d1 is not None and d2 is not None
    d = d1 - d2
    t = Affine.translation(d)
    return other.apply_transform(t)


def beside_snug(self: Diagram, other: Diagram, direction: V2) -> Diagram:
    return atop(self, juxtapose_snug(self, other, direction))


def juxtapose(self: Diagram, other: Diagram, direction: V2) -> Diagram:
    """Given two diagrams ``a`` and ``b``, ``a.juxtapose(b, v)``
    places ``b`` to touch ``a`` along angle ve .

    Args:
        self (Diagram): Diagram object.
        other (Diagram): Another diagram object.
        direction (V2): (Normalized) vector angle to juxtapose

    Returns:
        Diagram: Repositioned ``b`` diagram
    """
    envelope1 = self.get_envelope()
    envelope2 = other.get_envelope()
    d = envelope1.envelope_v(direction) - envelope2.envelope_v(-direction)
    t = Affine.translation(d)
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
    t = Affine.translation(envelope1.center)
    new_envelope = envelope1 + (t * envelope2)
    return self.compose(new_envelope, other.apply_transform(t))
