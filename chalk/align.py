from chalk.transform import V2, Affine, origin, unit_x, unit_y
from chalk.types import Diagram

# Functions mirroring Diagrams.Align and Diagrams.2d.Align


def align_to(self: Diagram, v: V2) -> Diagram:
    envelope = self.get_envelope()
    t = Affine.translation(-envelope.envelope_v(v))
    return self.apply_transform(t)


def snug(self: Diagram, v: V2) -> Diagram:
    "Align based on the trace."
    trace = self.get_trace()
    d = trace.trace_v(origin, v)
    assert d is not None
    t = Affine.translation(-d)
    return self.apply_transform(t)


def center(self: Diagram) -> Diagram:
    return self.center_xy()


# 2D versions


def align_t(self: Diagram) -> Diagram:
    return align_to(self, -unit_y)


def align_b(self: Diagram) -> Diagram:
    return align_to(self, unit_y)


def align_r(self: Diagram) -> Diagram:
    return align_to(self, unit_x)


def align_l(self: Diagram) -> Diagram:
    return align_to(self, -unit_x)


def align_tl(self: Diagram) -> Diagram:
    return align_l(align_t(self))


def align_br(self: Diagram) -> Diagram:
    return align_r(align_b(self))


def align_tr(self: Diagram) -> Diagram:
    return align_r(align_t(self))


def align_bl(self: Diagram) -> Diagram:
    return align_l(align_b(self))


def center_xy(self: Diagram) -> Diagram:
    envelope = self.get_envelope()
    if envelope.is_empty:
        return self
    t = Affine.translation(-envelope.center)
    return self.apply_transform(t)


def scale_uniform_to_x(self: Diagram, x: float) -> Diagram:
    """Apply uniform scaling along the x-axis.

    Args:
        self (Diagram): Diagram object.
        x (float): Amount of scaling along the x-axis.

    Returns:
        Diagram: A diagram object.
    """
    envelope = self.get_envelope()
    if envelope.is_empty:
        return self
    α = x / envelope.width
    return self.scale(α)


def scale_uniform_to_y(self: Diagram, y: float) -> Diagram:
    """Apply uniform scaling along the y-axis.

    Args:
        self (Diagram): Diagram object.
        y (float): Amount of scaling along the y-axis.

    Returns:
        Diagram: A diagram object.
    """
    envelope = self.get_envelope()
    if envelope.is_empty:
        return self
    α = y / envelope.height
    return self.scale(α)
