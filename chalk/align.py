from chalk.transform import V2, Affine, origin, unit_x, unit_y
from chalk.types import Diagram


def align(self: Diagram, v: V2) -> Diagram:
    envelope = self.get_envelope()
    t = Affine.translation(-envelope.envelope_v(v))
    return self.apply_transform(t)


def align_t(self: Diagram) -> Diagram:
    return align(self, -unit_y)


def align_b(self: Diagram) -> Diagram:
    return align(self, unit_y)


def align_r(self: Diagram) -> Diagram:
    return align(self, unit_x)


def align_l(self: Diagram) -> Diagram:
    return align(self, -unit_x)


def align_tl(self: Diagram) -> Diagram:
    return align_l(align_t(self))


def align_br(self: Diagram) -> Diagram:
    return align_r(align_b(self))


def align_tr(self: Diagram) -> Diagram:
    return align_r(align_t(self))


def align_bl(self: Diagram) -> Diagram:
    return align_l(align_b(self))


def snug(self: Diagram, v: V2) -> Diagram:
    "Align based on the trace."
    trace = self.get_trace()
    d = trace.trace_v(origin, v)
    assert d is not None
    t = Affine.translation(-d)
    return self.apply_transform(t)


def center_xy(self: Diagram) -> Diagram:
    envelope = self.get_envelope()
    if envelope.is_empty:
        return self
    t = Affine.translation(-envelope.center)
    return self.apply_transform(t)


def with_envelope(self: Diagram, other: Diagram) -> Diagram:
    return self.compose(other.get_envelope())
