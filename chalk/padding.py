from chalk.envelope import Envelope
from chalk.transform import V2
from chalk.types import Diagram

# def pad_l(self, extra: float) -> Diagram:
#     """Add outward directed left-side padding for
#     a diagram. This padding is applied **only** on
#     the **left** side.

#     Args:
#         extra (float): Amount of padding to add.

#     Returns:
#         Diagram: A diagram object.
#     """
#     return self
#     envelope = self.get_envelope()
#     if envelope is None:
#         return self
#     tl, br = envelope.min_point, envelope.max_point
#     new_envelope = Envelope.from_points([P2(tl.x - extra, tl.y), br])
#     return Compose(new_envelope, self, Empty())

# def pad_t(self, extra: float) -> Diagram:
#     """Add outward directed top-side padding for
#     a diagram. This padding is applied **only** on
#     the **top** side.

#     Args:
#         extra (float): Amount of padding to add.

#     Returns:
#         Diagram: A diagram object.
#     """
#     return self
#     envelope = self.get_envelope()
#     if envelope is None:
#         return self
#     tl, br = envelope.min_point, envelope.max_point
#     new_envelope = Envelope.from_points([P2(tl.x, tl.y - extra), br])
#     return Compose(new_envelope, self, Empty())

# def pad_r(self, extra: float) -> Diagram:
#     """Add outward directed right-side padding for
#     a diagram. This padding is applied **only** on
#     the **right** side.

#     Args:
#         extra (float): Amount of padding to add.

#     Returns:
#         Diagram: A diagram object.
#     """
#     return self
#     envelope = self.get_envelope()
#     if envelope is None:
#         return self
#     tl, br = envelope.min_point, envelope.max_point
#     new_envelope = Envelope.from_points([tl, P2(br.x + extra, br.y)])
#     return Compose(new_envelope, self, Empty())

# def pad_b(self, extra: float) -> Diagram:
#     """Add outward directed bottom-side padding for
#     a diagram. This padding is applied **only** on
#     the **bottom** side.

#     Args:
#         extra (float): Amount of padding to add.

#     Returns:
#         Diagram: A diagram object.
#     """
#     return self
#     envelope = self.get_envelope()
#     if envelope is None:
#         return self
#     tl, br = envelope.min_point, envelope.max_point
#     new_envelope = Envelope.from_points([tl, P2(br.x, br.y + extra)])
#     return Compose(new_envelope, self, Empty())


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
