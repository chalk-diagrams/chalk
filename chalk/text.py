from dataclasses import dataclass
from typing import Optional

from svgwrite import Drawing
from svgwrite.base import BaseElement

from chalk.shape import Shape
from chalk.style import Style
from chalk.transform import P2, BoundingBox, origin
from chalk.types import PyCairoContext, PyLatex, PyLatexElement, Diagram


@dataclass
class Text(Shape):
    """Text class."""

    text: str
    font_size: Optional[float]

    def get_bounding_box(self) -> BoundingBox:
        # Text doesn't have a bounding box since we can't accurately know
        # its size for all backends.
        eps = 1e-4
        self.bb = BoundingBox([origin, origin + P2(eps, eps)])
        return self.bb

    def render(self, ctx: PyCairoContext, style: Style) -> None:
        ctx.select_font_face("sans-serif")
        if self.font_size is not None:
            ctx.set_font_size(self.font_size)
        extents = ctx.text_extents(self.text)

        ctx.move_to(-(extents.width / 2), (extents.height / 2))
        ctx.text_path(self.text)

    def render_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        dx = -(self.bb.width / 2)
        return dwg.text(
            self.text,
            transform=f"translate({dx}, 0)",
            style=f"""text-align:center; text-anchor:middle; dominant-baseline:middle;
                      font-family:sans-serif; font-weight: bold;
                      font-size:{self.font_size}px;
                      vector-effect: non-scaling-stroke;""",
        )

    def render_tikz(self, pylatex: PyLatex, style: Style) -> PyLatexElement:
        opts = {}
        opts["font"] = "\\small\\sffamily"
        opts["scale"] = str(
            3.5 * (1 if self.font_size is None else self.font_size)
        )
        styles = style.to_tikz(pylatex)
        if styles["fill"] is not None:
            opts["text"] = styles["fill"]
        return pylatex.TikZNode(
            text=self.text,
            # Scale parameters based on observations
            options=pylatex.TikZOptions(**opts),
        )


def text(t: str, size: Optional[float]) -> Diagram:
    """
    Draw some text.

    Args:
       t (str): The text string.
       size (Optional[float]): Size of the text.

    Returns:
       Diagram

    """
    from chalk.core import Primitive

    return Primitive.from_shape(Text(t, font_size=size))
