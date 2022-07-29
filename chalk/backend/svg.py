from typing import Any, List, Optional, Tuple

import svgwrite
from svgwrite import Drawing
from svgwrite.base import BaseElement

from chalk import transform as tx

from chalk.style import Style
from chalk.transform import unit_x, unit_y
from chalk.visitor import DiagramVisitor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chalk.core import Primitive


class ToSVG(DiagramVisitor[BaseElement]):
    def visit_primitive(self, diagram: "Primitive", dwg: Drawing, style: Style) -> BaseElement:
        """Convert a diagram to SVG image."""
        style = diagram.style.merge(style)
        style_svg = style.to_svg()
        transform = tx.to_svg(diagram.transform)
        inner = diagram.shape.render_svg(dwg, style)
        if not style_svg and not transform:
            return inner
        else:
            if not style_svg:
                style_svg = ";"
            g = dwg.g(transform=transform, style=style_svg)
            g.add(inner)
            return g

    def visit_empty(self, diagram, dwg: Drawing, style: Style) -> BaseElement:
        """Converts to SVG image."""
        return dwg.g()

    def visit_compose(self, diagram, dwg: Drawing, style: Style) -> BaseElement:
        g = dwg.g()
        g.add(diagram.diagram1.accept(self, dwg=dwg, style=style))
        g.add(diagram.diagram2.accept(self, dwg=dwg, style=style))
        return g

    def visit_apply_transform(self, diagram, dwg: Drawing, style: Style) -> BaseElement:
        g = dwg.g(transform=tx.to_svg(diagram.transform))
        g.add(diagram.diagram.accept(self, dwg=dwg, style=style))
        return g

    def visit_apply_style(self, diagram, dwg: Drawing, style: Style) -> BaseElement:
        return diagram.diagram.accept(self, dwg=dwg, style=diagram.style.merge(style))

    def visit_apply_name(self, diagram, dwg: Drawing, style: Style) -> BaseElement:
        g = dwg.g()
        g.add(diagram.diagram.accept(self, dwg=dwg, style=style))
        return g


def render(
    self, path: str, height: int = 128, width: Optional[int] = None
) -> None:
    """Render the diagram to an SVG file.

    Args:
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
    dwg = svgwrite.Drawing(
        path,
        size=(width, height),
    )

    outer = dwg.g(
        style="fill:white;",
    )
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
    outer.add(s.accept(ToSVG(), dwg=dwg, style=style))
    dwg.save()
