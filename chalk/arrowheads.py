from dataclasses import dataclass

from colour import Color
from svgwrite import Drawing
from svgwrite.base import BaseElement

from chalk.path import Path
from chalk.shape import Shape, render_cairo_prims
from chalk.style import Style
from chalk.transform import P2, BoundingBox, origin
from chalk.types import Diagram, PyCairoContext, PyLatex, PyLatexElement

black = Color("black")


def tri() -> Diagram:
    from chalk.core import Primitive

    return (
        Primitive.from_shape(
            Path.from_points([(1.0, 0), (0.0, -1.0), (-1.0, 0), (1.0, 0)])
        )
        .rotate_by(-0.25)
        .fill_color(Color("black"))
        .align_r()
        .line_width(0)
    )


def dart(cut: float = 0.2) -> Diagram:
    from chalk.core import Primitive

    return (
        Primitive.from_shape(
            Path.from_points(
                [
                    (0, -cut),
                    (1.0, cut),
                    (0.0, -1.0 - cut),
                    (-1.0, +cut),
                    (0, -cut),
                ]
            )
        )
        .rotate_by(-0.25)
        .fill_color(Color("black"))
        .align_r()
        .line_width(0)
    )


@dataclass
class ArrowHead(Shape):
    """Arrow Head."""

    arrow_shape: Diagram

    def get_bounding_box(self) -> BoundingBox:
        # Arrow head don't have a bounding box since we can't accurately know
        # the size until rendering
        eps = 1e-4
        self.bb = BoundingBox([origin, origin + P2(eps, eps)])
        return self.bb

    def render(self, ctx: PyCairoContext, style: Style) -> None:
        assert style.output_size
        scale = 0.01 * (15 / 500) * style.output_size
        render_cairo_prims(self.arrow_shape.scale(scale), ctx, style)

    def render_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        assert style.output_size
        scale = 0.01 * (15 / 500) * style.output_size
        return self.arrow_shape.scale(scale).to_svg(dwg, style)

    def render_tikz(self, p: PyLatex, style: Style) -> PyLatexElement:
        assert style.output_size
        scale = 0.01 * 3 * (15 / 500) * style.output_size
        s = p.TikZScope()
        for inner in self.arrow_shape.scale(scale).to_tikz(p, style):
            s.append(inner)
        return
