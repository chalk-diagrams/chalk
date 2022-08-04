from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import svgwrite
from svgwrite import Drawing
from svgwrite.base import BaseElement

from chalk import transform as tx
from chalk.style import Style
from chalk.transform import unit_x, unit_y
from chalk.types import Diagram
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import (
        ApplyName,
        ApplyStyle,
        ApplyTransform,
        Compose,
        Empty,
        Primitive,
    )


EMTPY_STYLE = Style.empty()


class ToSVG(DiagramVisitor[BaseElement]):
    def __init__(self, dwg: Drawing):
        self.dwg = dwg

    def visit_primitive(
        self, diagram: Primitive, style: Style = EMTPY_STYLE
    ) -> BaseElement:
        style_new = diagram.style.merge(style)
        style_svg = style_new.to_svg()
        transform = tx.to_svg(diagram.transform)
        inner = diagram.shape.render_svg(self.dwg, style_new)
        if not style_svg and not transform:
            return inner
        else:
            if not style_svg:
                style_svg = ";"
            g = self.dwg.g(transform=transform, style=style_svg)
            g.add(inner)
            return g

    def visit_empty(
        self, diagram: Empty, style: Style = EMTPY_STYLE
    ) -> BaseElement:
        return self.dwg.g()

    def visit_compose(
        self, diagram: Compose, style: Style = EMTPY_STYLE
    ) -> BaseElement:
        g = self.dwg.g()
        g.add(diagram.diagram1.accept(self, style=style))
        g.add(diagram.diagram2.accept(self, style=style))
        return g

    def visit_apply_transform(
        self, diagram: ApplyTransform, style: Style = EMTPY_STYLE
    ) -> BaseElement:
        g = self.dwg.g(transform=tx.to_svg(diagram.transform))
        g.add(diagram.diagram.accept(self, style=style))
        return g

    def visit_apply_style(
        self, diagram: ApplyStyle, style: Style = EMTPY_STYLE
    ) -> BaseElement:
        return diagram.diagram.accept(self, style=diagram.style.merge(style))

    def visit_apply_name(
        self, diagram: ApplyName, style: Style = EMTPY_STYLE
    ) -> BaseElement:
        g = self.dwg.g()
        g.add(diagram.diagram.accept(self, style=style))
        return g


def to_svg(self: Diagram, dwg: Drawing, style: Style) -> BaseElement:
    return self.accept(ToSVG(dwg), style=style)


def render(
    self: Diagram, path: str, height: int = 128, width: Optional[int] = None
) -> None:
    """Render the diagram to an SVG file.

    Args:
        self (Diagram): Given ``Diagram`` instance.
        path (str): Path of the .svg file.
        height (int, optional): Height of the rendered image.
                                Defaults to 128.
        width (Optional[int], optional): Width of the rendered image.
                                         Defaults to None.
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
    style = Style.root(output_size=max(height, width))
    outer.add(to_svg(s, dwg, style))
    dwg.save()
