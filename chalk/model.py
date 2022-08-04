from colour import Color

from chalk.path import Path
from chalk.shapes import circle
from chalk.transform import V2, Vec2Array, origin
from chalk.types import Diagram


def show_origin(self: Diagram) -> Diagram:
    "Add a red dot at the origin of a diagram for debugging."
    from chalk.core import Primitive

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
    from chalk.core import Primitive

    self.show_origin()
    envelope = self.get_envelope()
    if envelope.is_empty:
        return self
    outer: Diagram = (
        Primitive.from_shape(Path(envelope.to_path(angle)))
        .fill_opacity(0)
        .line_color(Color("red"))
    )
    for segment in envelope.to_segments(angle):
        outer = outer + Primitive.from_shape(Path(segment)).line_color(
            Color("red")
        ).dashing([0.01, 0.01], 0)

    new = self + outer
    if phantom:
        new.with_envelope(self)
    return new


# show_label


def show_beside(self: Diagram, other: Diagram, direction: V2) -> Diagram:
    "Add blue normal line to show placement of combination."
    from chalk.core import Primitive

    envelope1 = self.get_envelope()
    envelope2 = other.get_envelope()
    v1 = envelope1.envelope_v(direction)
    one: Diagram = (
        Primitive.from_shape(Path(Vec2Array([origin, v1])))
        .line_color(Color("red"))
        .dashing([0.01, 0.01], 0)
        .line_width(0.01)
    )
    v2 = envelope2.envelope_v(-direction)
    two: Diagram = (
        Primitive.from_shape(Path(Vec2Array([origin, v2])))
        .line_color(Color("red"))
        .dashing([0.01, 0.01], 0)
        .line_width(0.01)
    )
    split: Diagram = (
        Primitive.from_shape(
            Path(
                Vec2Array(
                    [
                        v1 + direction.perpendicular(),
                        v1 - direction.perpendicular(),
                    ]
                )
            )
        )
        .line_color(Color("blue"))
        .line_width(0.02)
    )
    one = (self.show_origin() + one + split).with_envelope(self)
    two = (other.show_origin() + two).with_envelope(other)
    return one.beside(two, direction)
