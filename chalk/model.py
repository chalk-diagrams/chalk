from colour import Color

from chalk.combinators import concat
from chalk.shapes import Path, circle
from chalk.shapes.segment import seg
from chalk.transform import V2, origin
from chalk.types import Diagram


def show_origin(self: Diagram) -> Diagram:
    "Add a red dot at the origin of a diagram for debugging."

    envelope = self.get_envelope()
    if envelope.is_empty:
        return self
    origin_size = min(envelope.height, envelope.width) / 50
    origin = circle(origin_size).line_color(Color("red"))
    return self + origin


def show_envelope(
    self: Diagram, phantom: bool = False, angle: int = 45
) -> Diagram:
    """Add red envelope to diagram for debugging.

    Args:
        self (Diagram) : Diagram
        phantom (bool): Don't include debugging in the envelope
        angle (int): Angle increment to show debugging lines.

    Returns:
        Diagram
    """

    self.show_origin()
    envelope = self.get_envelope()
    if envelope.is_empty:
        return self
    outer: Diagram = (
        Path.from_points(list(envelope.to_path(angle)))
        .stroke()
        .fill_opacity(0)
        .line_color(Color("red"))
    )
    outer += (
        concat([seg(y).stroke() for (x, y) in envelope.to_segments(angle)])
        .line_color(Color("red"))
        .dashing([0.01, 0.01], 0)
    )

    new = self + outer
    if phantom:
        new.with_envelope(self)
    return new


# show_label


def show_beside(self: Diagram, other: Diagram, direction: V2) -> Diagram:
    "Add blue normal line to show placement of combination."

    envelope1 = self.get_envelope()
    envelope2 = other.get_envelope()
    v1 = envelope1.envelope_v(direction)
    one: Diagram = (
        Path.from_points([origin, v1])
        .stroke()
        .line_color(Color("red"))
        .dashing([0.01, 0.01], 0)
        .line_width(0.01)
    )
    v2 = envelope2.envelope_v(-direction)
    two: Diagram = (
        Path.from_points([origin, v2])
        .stroke()
        .line_color(Color("red"))
        .dashing([0.01, 0.01], 0)
        .line_width(0.01)
    )
    split: Diagram = (
        Path.from_points(
            [
                v1 + direction.perpendicular(),
                v1 - direction.perpendicular(),
            ]
        )
        .stroke()
        .line_color(Color("blue"))
        .line_width(0.02)
    )
    one = (self.show_origin() + one + split).with_envelope(self)
    two = (other.show_origin() + two).with_envelope(other)
    return one.beside(two, direction)
