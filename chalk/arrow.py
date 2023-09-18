from dataclasses import dataclass, field, replace
from typing import Dict, List, Literal, Optional, Tuple

from colour import Color

from planar.py import Ray

from chalk.shapes import ArcSegment, ArrowHead, arc_seg, dart
from chalk.shapes.segment import ray_ray_intersection
from chalk.style import Style
from chalk.subdiagram import Name, Subdiagram
from chalk.trail import Trail
from chalk.transform import P2, V2, unit_x, unit_y
from chalk.types import Diagram

black = Color("black")
# arrow heads


@dataclass
class ArrowOpts:
    head_style: Style = field(default_factory=Style)
    head_pad: float = 0.0
    tail_pad: float = 0.0
    head_arrow: Optional[Diagram] = None
    shaft_style: Style = field(default_factory=Style)
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

        ps = tr1.trace_p(midpoint, -v)
        pe = tr2.trace_p(midpoint, v)

        assert ps is not None, "Cannot connect"
        assert pe is not None, "Cannot connect"

        return dia + arrow_between(ps, pe, style)

    return self.with_names([name1, name2], f)


ElbowType = Literal["hv", "vh"]


def connect_outside_elbow(
    self: Diagram,
    name1: Name,
    name2: Name,
    elbow_type: ElbowType,
    style: ArrowOpts = ArrowOpts(),
) -> Diagram:
    from chalk.core import empty

    directions = {
        "hv": (unit_x, unit_y),
        "vh": (unit_y, unit_x),
    }

    def f(subs: List[Subdiagram], dia: Diagram) -> Diagram:
        sub1, sub2 = subs

        loc1 = sub1.get_location()
        loc2 = sub2.get_location()

        dir1, dir2 = directions[elbow_type]
        ray1 = Ray(loc1, dir1)
        ray2 = Ray(loc2, dir2)

        t = ray_ray_intersection(ray1, ray2)

        assert t is not None, "Cannot connect"
        t1, _ = t
        pm = ray1.anchor + t1 * ray1.direction

        tr1 = sub1.get_trace()
        tr2 = sub2.get_trace()

        if loc1 != pm:
            dir1 = loc1 - pm

        if loc2 != pm:
            dir2 = loc2 - pm

        ps = tr1.trace_p(loc1, dir1)
        pe = tr2.trace_p(loc2, dir2)

        assert ps is not None, "Cannot connect"
        assert pe is not None, "Cannot connect"

        style1 = replace(style, head_arrow=empty())
        return dia + arrow_between(ps, pm, style1) + arrow_between(pm, pe, style)

    return self.with_names([name1, name2], f)


def connect_perim(
    self: Diagram,
    name1: Name,
    name2: Name,
    v1: V2,
    v2: V2,
    style: ArrowOpts = ArrowOpts(),
) -> Diagram:
    def f(subs: List[Subdiagram], dia: Diagram) -> Diagram:
        sub1, sub2 = subs

        loc1 = sub1.get_location()
        loc2 = sub2.get_location()

        tr1 = sub1.get_trace()
        tr2 = sub2.get_trace()

        ps = tr1.max_trace_p(loc1, v1)
        pe = tr2.max_trace_p(loc2, v2)

        assert ps is not None, "Cannot connect"
        assert pe is not None, "Cannot connect"

        return dia + arrow_between(ps, pe, style)

    return self.with_names([name1, name2], f)


# Arrow primitives


def arrow(length: float, style: ArrowOpts = ArrowOpts()) -> Diagram:
    from chalk.core import Primitive

    if style.head_arrow is None:
        arrow: Diagram = Primitive.from_shape(ArrowHead(dart()))
    else:
        arrow = style.head_arrow
    arrow = arrow._style(style.head_style)
    t = style.tail_pad
    l_adj = length - style.head_pad - t
    if style.trail is None:
        segment = arc_seg(P2(l_adj, 0), style.arc_height)
        shaft = segment.stroke()
        if isinstance(segment.segments[-1], ArcSegment):
            seg = segment.segments[-1]
            tan = -(seg.q - seg.center.reflect_y()).perpendicular()  # type: ignore
            φ = tan.angle
            arrow = arrow.rotate(φ)
            if style.arc_height < 0:
                arrow = arrow.rotate(180)
    else:
        shaft = style.trail.stroke().scale_uniform_to_x(l_adj).fill_opacity(0)

        if isinstance(style.trail.segments[-1], ArcSegment):
            arrow = arrow.rotate(-style.trail.segments[-1].angle)

    return shaft._style(style.shaft_style).translate_by(
        t * unit_x
    ) + arrow.translate_by((l_adj + t) * unit_x)


def arrow_v(vec: V2, style: ArrowOpts = ArrowOpts()) -> Diagram:
    arr = arrow(vec.length, style)
    return arr.rotate(-vec.angle)


def arrow_at(base: P2, vec: V2, style: ArrowOpts = ArrowOpts()) -> Diagram:
    return arrow_v(vec, style).translate_by(base)


def arrow_between(start: P2, end: P2, style: ArrowOpts = ArrowOpts()) -> Diagram:
    return arrow_at(start, end - start, style)
