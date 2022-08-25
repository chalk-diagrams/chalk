from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Optional

import svgwrite
from svgwrite import Drawing
from svgwrite.base import BaseElement
from svgwrite.shapes import Rect

from chalk import transform as tx
from chalk.shapes import (
    ArcSegment,
    ArrowHead,
    Image,
    Latex,
    Path,
    Segment,
    SegmentLike,
    Spacer,
    Text,
)
from chalk.style import Style
from chalk.transform import P2, unit_x, unit_y
from chalk.types import Diagram
from chalk.visitor import DiagramVisitor, ShapeVisitor

if TYPE_CHECKING:
    from chalk.core import (
        ApplyName,
        ApplyStyle,
        ApplyTransform,
        Compose,
        Empty,
        Primitive,
    )


EMPTY_STYLE = Style.empty()


def tx_to_svg(affine: tx.Affine) -> str:
    def convert(
        a: float, b: float, c: float, d: float, e: float, f: float
    ) -> str:
        return f"matrix({a}, {d}, {b}, {e}, {c}, {f})"

    return convert(*affine[:6])


class Raw(Rect):  # type: ignore
    """Shape class.

    A fake SVG node for importing latex.
    """

    def __init__(self, st: str):
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        self.xml = ET.fromstring(st)

    def get_xml(self) -> ET.Element:
        return self.xml


class ToSVG(DiagramVisitor[BaseElement]):
    def __init__(self, dwg: Drawing):
        self.dwg = dwg
        self.shape_renderer = ToSVGShape(dwg)

    def visit_primitive(
        self, diagram: Primitive, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        style_new = diagram.style.merge(style)
        style_svg = style_new.to_svg()
        transform = tx_to_svg(diagram.transform)
        inner = diagram.shape.accept(self.shape_renderer, style=style_new)
        if not style_svg and not transform:
            return inner
        else:
            if not style_svg:
                style_svg = ";"
            g = self.dwg.g(transform=transform, style=style_svg)
            g.add(inner)
            return g

    def visit_empty(
        self, diagram: Empty, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        return self.dwg.g()

    def visit_compose(
        self, diagram: Compose, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        g = self.dwg.g()
        g.add(diagram.diagram1.accept(self, style=style))
        g.add(diagram.diagram2.accept(self, style=style))
        return g

    def visit_apply_transform(
        self, diagram: ApplyTransform, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        g = self.dwg.g(transform=tx_to_svg(diagram.transform))
        g.add(diagram.diagram.accept(self, style=style))
        return g

    def visit_apply_style(
        self, diagram: ApplyStyle, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        return diagram.diagram.accept(self, style=diagram.style.merge(style))

    def visit_apply_name(
        self, diagram: ApplyName, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        g = self.dwg.g()
        g.add(diagram.diagram.accept(self, style=style))
        return g


class ToSVGShape(ShapeVisitor[BaseElement]):
    def __init__(self, dwg: Drawing):
        self.dwg = dwg

    def render_segment(self, seg: SegmentLike, p: P2) -> str:
        q = seg.q + p
        if isinstance(seg, Segment):
            return f"L {q.x} {q.y}"
        elif isinstance(seg, ArcSegment):
            "https://www.w3.org/TR/SVG/implnote.html#ArcConversionCenterToEndpoint"
            f_A = 1 if abs(seg.dangle) > 180 else 0
            det: float = seg.t.determinant  # type: ignore
            f_S = 1 if det * seg.dangle > 0 else 0
            return f"A {seg.r_x} {seg.r_y} {seg.rot} {f_A} {f_S} {q.x} {q.y}"

    def visit_path(
        self, path: Path, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        extra_style = ""
        if not path.loc_trails[0].trail.closed:
            extra_style = "fill:none;"
        line = self.dwg.path(
            style="vector-effect: non-scaling-stroke;" + extra_style,
        )
        for loc_trail in path.loc_trails:
            p = loc_trail.location
            line.push(f"M {p.x} {p.y}")
            for i, (seg, p) in enumerate(loc_trail.located_segments()):
                line.push(self.render_segment(seg, p))
            if loc_trail.trail.closed:
                line.push("Z")
        return line

    def visit_latex(
        self, shape: Latex, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        dx, dy = -shape.width / 2, -shape.height / 2
        g = self.dwg.g(transform=f"scale(0.05) translate({dx} {dy})")
        g.add(Raw(shape.content))
        return g

    def visit_text(
        self, shape: Text, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        dx = -(shape.get_bounding_box().width / 2)
        return self.dwg.text(
            shape.text,
            transform=f"translate({dx}, 0)",
            style=f"""text-align:center; text-anchor:middle; dominant-baseline:middle;
                      font-family:sans-serif; font-weight: bold;
                      font-size:{shape.font_size}px;
                      vector-effect: non-scaling-stroke;""",
        )

        raise NotImplementedError("Text is not implemented")

    def visit_spacer(
        self, shape: Spacer, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        return self.dwg.g()

    def visit_arrowhead(
        self, shape: ArrowHead, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        assert style.output_size
        scale = 0.01 * (15 / 500) * style.output_size
        return to_svg(shape.arrow_shape.scale(scale), self.dwg, style)

    def visit_image(
        self, shape: Image, style: Style = EMPTY_STYLE
    ) -> BaseElement:
        dx = -shape.width / 2
        dy = -shape.height / 2
        return self.dwg.image(
            href=shape.url_path, transform=f"translate({dx}, {dy})"
        )


def to_svg(self: Diagram, dwg: Drawing, style: Style) -> BaseElement:
    return self.accept(ToSVG(dwg), style=style)


def render(
    self: Diagram,
    path: str,
    height: int = 128,
    width: Optional[int] = None,
    draw_height: Optional[int] = None,
) -> None:
    """Render the diagram to an SVG file.

    Args:
        self (Diagram): Given ``Diagram`` instance.
        path (str): Path of the .svg file.
        height (int, optional): Height of the rendered image.
                                Defaults to 128.
        width (Optional[int], optional): Width of the rendered image.
                                         Defaults to None.
        draw_height (Optional[int], optional): Override the height for
                                               line width.

    """
    pad = 0.05
    envelope = self.get_envelope()

    # infer width to preserve aspect ratio
    assert envelope is not None
    width = width or int(height * envelope.width / envelope.height)

    # determine scale to fit the largest axis in the target frame size
    if envelope.width - width <= envelope.height - height:
        α = height / ((1 + pad) * envelope.height)
    else:
        α = width / ((1 + pad) * envelope.width)

    dwg = svgwrite.Drawing(path, size=(width, height))

    outer = dwg.g(style="fill:white;")
    # Arrow marker
    marker = dwg.marker(
        id="arrow", refX=5.0, refY=1.7, size=(5, 3.5), orient="auto"
    )
    marker.add(dwg.polygon([(0, 0), (5, 1.75), (0, 3.5)]))
    dwg.defs.add(marker)

    dwg.add(outer)
    s = self.center_xy().pad(1 + pad).scale(α)
    e = s.get_envelope()
    assert e is not None
    s = s.translate(e(-unit_x), e(-unit_y))
    if draw_height is None:
        draw_height = max(height, width)
    style = Style.root(output_size=draw_height)
    outer.add(to_svg(s, dwg, style))
    dwg.save()
