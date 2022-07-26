from chalk.transform import V2, Affine, origin, unit_y
from chalk.types import Diagram


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


def atop(self: Diagram, other: Diagram) -> Diagram:
    envelope1 = self.get_envelope()
    envelope2 = other.get_envelope()
    new_envelope = envelope1 + envelope2
    return self.compose(new_envelope, other)


def above(self: Diagram, other: Diagram) -> Diagram:
    return beside(self, other, unit_y)


def beside(self: Diagram, other: Diagram, direction: V2) -> Diagram:
    return atop(self, juxtapose(self, other, direction))


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
