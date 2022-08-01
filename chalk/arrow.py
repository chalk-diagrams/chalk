import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from colour import Color

from chalk.arrowheads import ArrowHead, dart
from chalk.path import Path
from chalk.shape import Arc
from chalk.style import Style
from chalk.trail import Trail
from chalk.transform import P2, V2, unit_x
from chalk.types import Diagram

black = Color("black")
# arrow heads


@dataclass
class ArrowOpts:
    head_style: Style = Style()
    head_pad: float = 0.0
    tail_pad: float = 0.0
    head_arrow: Optional[Diagram] = None
    shaft_style: Style = Style()
    trail: Optional[Trail] = None
    arc_height: float = 0.0


# Arrow connections.


def connect(
    self: Diagram, name1: str, name2: str, style: ArrowOpts = ArrowOpts()
) -> Diagram:
    bb1 = self.get_subdiagram_envelope(name1)
    bb2 = self.get_subdiagram_envelope(name2)
    return self + arrow_between(bb1.center, bb2.center, style)


def connect_outside(
    self: Diagram, name1: str, name2: str, style: ArrowOpts = ArrowOpts()
) -> Diagram:
    env1 = self.get_subdiagram_envelope(name1)
    env2 = self.get_subdiagram_envelope(name2)

    tr1 = self.get_subdiagram_trace(name1)
    tr2 = self.get_subdiagram_trace(name2)

    v = env2.center - env1.center
    midpoint = env1.center + v / 2

    ps = tr1.trace_p(midpoint, -v)
    pe = tr2.trace_p(midpoint, v)

    return self + arrow_between(ps, pe, style)


def connect_perim(
    self: Diagram,
    name1: str,
    name2: str,
    v1: V2,
    v2: V2,
    style: ArrowOpts = ArrowOpts(),
) -> Diagram:
    env1 = self.get_subdiagram_envelope(name1)
    env2 = self.get_subdiagram_envelope(name2)

    tr1 = self.get_subdiagram_trace(name1)
    tr2 = self.get_subdiagram_trace(name2)

    ps = tr1.max_trace_p(env1.center, v1)
    pe = tr2.max_trace_p(env2.center, v2)

    return self + arrow_between(ps, pe, style)


# Arrow primitiv
def make_path(
    coords: Union[List[Tuple[float, float]], List[P2]], arrow: bool = False
) -> Diagram:
    from chalk.core import Primitive

    if not coords or isinstance(coords[0], P2):
        return Primitive.from_shape(Path.from_points(coords))
    else:
        return Primitive.from_shape(Path.from_list_of_tuples(coords, arrow))


def unit_arc_between(d: float, height: float) -> Tuple[Diagram, float]:
    from chalk.core import Primitive

    h = abs(height)
    θ = 0.0
    if h < 1e-6:
        # Draw a line if the height is too small
        shape: Diagram = make_path([(0, 0), (d, 0)])
    else:
        # Determine the arc's angle θ and its radius r
        θ = math.acos((d**2 - 4.0 * h**2) / (d**2 + 4.0 * h**2))
        r = d / (2 * math.sin(θ))

        if height > 0:
            # bend left
            φ = -math.pi / 2
            dy = r - h
        else:
            # bend right
            φ = +math.pi / 2
            dy = h - r
        Primitive.from_shape(Arc(r, -θ, θ))
        shape = (
            Primitive.from_shape(Arc(r, -θ, θ))
            .rotate_rad(-φ)
            .translate(d / 2, dy)
        )
    return shape, -θ if height > 0 else θ


def arrow(length: int, style: ArrowOpts = ArrowOpts()) -> Diagram:
    from chalk.core import Primitive

    if style.head_arrow is None:
        arrow: Diagram = Primitive.from_shape(ArrowHead(dart()))
    else:
        arrow = style.head_arrow
    arrow = arrow._style(style.head_style)
    t = style.tail_pad
    l_adj = length - style.head_pad - t
    if style.trail is None:
        shaft, φ = unit_arc_between(l_adj, style.arc_height)
        arrow = arrow.rotate_rad(φ)
    else:
        shaft = style.trail.stroke().scale_uniform_to_x(l_adj).fill_opacity(0)
        arrow = arrow.rotate(-style.trail.offsets[-1].angle)

    return shaft._style(style.shaft_style).translate_by(
        t * unit_x
    ) + arrow.translate_by((l_adj + t) * unit_x)


def arrow_v(vec: V2, style: ArrowOpts = ArrowOpts()) -> Diagram:
    arr = arrow(vec.length, style)
    return arr.rotate(-vec.angle)


def arrow_at(base: P2, vec: V2, style: ArrowOpts = ArrowOpts()) -> Diagram:
    return arrow_v(vec, style).translate_by(base)


def arrow_between(
    start: P2, end: P2, style: ArrowOpts = ArrowOpts()
) -> Diagram:
    return arrow_at(start, end - start, style)
