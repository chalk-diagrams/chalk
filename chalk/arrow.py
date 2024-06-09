from dataclasses import dataclass, field
from typing import List, Optional

from colour import Color

import chalk.transform as tx
from chalk.shapes import ArrowHead, arc_seg, dart
from chalk.style import StyleHolder
from chalk.subdiagram import Name, Subdiagram
from chalk.trail import Trail
from chalk.transform import P2_t, V2_t
from chalk.types import Diagram

black = Color("black")
# arrow heads


@dataclass
class ArrowOpts:
    head_style: StyleHolder = field(default_factory=StyleHolder.empty)
    head_pad: float = 0.0
    tail_pad: float = 0.0
    head_arrow: Optional[Diagram] = None
    shaft_style: StyleHolder = field(default_factory=StyleHolder.empty)
    trail: Optional[Trail] = None
    arc_height: float = 0.0


# Arrow connections.


def connect(
    self: Diagram, name1: Name, name2: Name, style: ArrowOpts = ArrowOpts()
) -> Diagram:
    def f(subs: List[Subdiagram], dia: Diagram) -> Diagram:
        sub1, sub2 = subs

        ps = sub1.get_location()
        pe = sub2.get_location()
        return dia + arrow_between(ps, pe, style)

    return self.with_names([name1, name2], f)


def connect_outside(
    self: Diagram, name1: Name, name2: Name, style: ArrowOpts = ArrowOpts()
) -> Diagram:
    def f(subs: List[Subdiagram], dia: Diagram) -> Diagram:
        sub1, sub2 = subs

        loc1 = sub1.get_location()
        loc2 = sub2.get_location()

        tr1 = sub1.get_trace()
        tr2 = sub2.get_trace()

        v = loc2 - loc1
        midpoint = loc1 + v / 2

        ps, m1 = tr1.trace_p(midpoint, -v)
        pe, m2 = tr2.trace_p(midpoint, v)

        assert m1.all(), "Cannot connect"
        assert m2.all(), "Cannot connect"

        return dia + arrow_between(ps, pe, style)

    return self.with_names([name1, name2], f)


def connect_perim(
    self: Diagram,
    name1: Name,
    name2: Name,
    v1: V2_t,
    v2: V2_t,
    style: ArrowOpts = ArrowOpts(),
) -> Diagram:
    def f(subs: List[Subdiagram], dia: Diagram) -> Diagram:
        sub1, sub2 = subs

        loc1 = sub1.get_location()
        loc2 = sub2.get_location()

        tr1 = sub1.get_trace()
        tr2 = sub2.get_trace()

        ps, m1 = tr1.max_trace_p(loc1, v1)
        pe, m2 = tr2.max_trace_p(loc2, v2)

        assert m1.all(), "Cannot connect"
        assert m2.all(), "Cannot connect"

        return dia + arrow_between(ps, pe, style)

    return self.with_names([name1, name2], f)


# Arrow primitives


def arrow(length: tx.Floating, style: ArrowOpts = ArrowOpts()) -> Diagram:
    from chalk.core import Primitive

    if style.head_arrow is None:
        arrow: Diagram = Primitive.from_shape(ArrowHead(dart()))
    else:
        arrow = style.head_arrow
    arrow = arrow._style(style.head_style)
    t = style.tail_pad
    l_adj = length - style.head_pad - t
    if style.trail is None:
        segment = arc_seg(tx.V2(l_adj, 0), style.arc_height + 1e-3)
        shaft = segment.stroke()
        if False:  # isinstance(segment.segments[-1], Segment):
            seg = segment.segments[-1]
            tan = -(seg.q - seg.center.reflect_y()).perpendicular()  # type: ignore
            φ = tan.angle
            arrow = arrow.rotate(φ)
            if style.arc_height < 0:
                arrow = arrow.rotate(180)
    else:
        shaft = style.trail.stroke().scale_uniform_to_x(l_adj).fill_opacity(0)

        if False:  # isinstance(style.trail.segments[-1], Segment):
            arrow = arrow.rotate(-style.trail.segments[-1].angle)

    return shaft._style(style.shaft_style).translate_by(
        t * tx.unit_x
    ) + arrow.translate_by((l_adj + t) * tx.unit_x)


def arrow_v(vec: V2_t, style: ArrowOpts = ArrowOpts()) -> Diagram:
    print("VEC", vec)
    arr = arrow(tx.length(vec), style)
    return arr.rotate(-tx.angle(vec))


def arrow_at(base: P2_t, vec: V2_t, style: ArrowOpts = ArrowOpts()) -> Diagram:
    return arrow_v(vec, style).translate_by(base)


def arrow_between(
    start: P2_t, end: P2_t, style: ArrowOpts = ArrowOpts()
) -> Diagram:
    return arrow_at(start, end - start, style)
