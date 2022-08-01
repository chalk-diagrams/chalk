import xml.etree.ElementTree as ET
from dataclasses import dataclass

from svgwrite import Drawing
from svgwrite.base import BaseElement

from chalk.shape import Rectangle, Shape
from chalk.style import Style
from chalk.transform import P2, BoundingBox, origin
from chalk.types import Diagram, PyCairoContext


class Raw(Rectangle):
    """Shape class.

    A fake SVG node for importing latex.
    """

    def __init__(self, st: str):
        self.xml = ET.fromstring(st)

    def get_xml(self) -> ET.Element:
        return self.xml


@dataclass
class Latex(Shape):
    """Latex class."""

    text: str

    def __post_init__(self) -> None:
        # Need to install latextools for this to run.
        import latextools

        # Border ensures no cropping.
        latex_eq = latextools.render_snippet(
            f"{self.text}",
            commands=[latextools.cmd.all_math],
            config=latextools.DocumentConfig(
                "standalone", {"crop=true,border=0.1cm"}
            ),
        )
        self.eq = latex_eq.as_svg()
        self.width = self.eq.width
        self.height = self.eq.height
        self.content = self.eq.content
        # From latextools Ensures no clash between multiple math statements
        id_prefix = f"embed-{hash(self.content)}-"
        self.content = (
            self.content.replace('id="', f'id="{id_prefix}')
            .replace('="url(#', f'="url(#{id_prefix}')
            .replace('xlink:href="#', f'xlink:href="#{id_prefix}')
        )

    def get_bounding_box(self) -> BoundingBox:
        left = origin.x - self.width / 2
        top = origin.y - self.height / 2
        tl = P2(left, top)
        br = P2(left + self.width, top + self.height)
        return BoundingBox(tl, br).scale(0.05)

    def render(self, ctx: PyCairoContext, style: Style) -> None:
        raise NotImplementedError

    def render_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        dx, dy = -self.width / 2, -self.height / 2
        g = dwg.g(transform=f"scale(0.05) translate({dx} {dy})")
        g.add(Raw(self.content))
        return g


def latex(t: str) -> Diagram:
    from chalk.core import Primitive

    return Primitive.from_shape(Latex(t))
