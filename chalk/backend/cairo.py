from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import chalk.transform as tx
from chalk.shapes import (
    ArrowHead,
    Image,
    Latex,
    Path,
    Segment,
    Spacer,
    Text,
    from_pil,
)
from chalk.style import Style, StyleHolder
from chalk.transform import Affine
from chalk.types import Diagram
from chalk.visitor import ShapeVisitor

if TYPE_CHECKING:
    from chalk.core import Primitive


Ident = tx.ident
PyCairoContext = Any
EMPTY_STYLE = StyleHolder.empty()


def tx_to_cairo(affine: Affine) -> Any:
    import cairo

    def convert(a, b, c, d, e, f):  # type: ignore
        return cairo.Matrix(a, d, b, e, c, f)  # type: ignore

    return convert(*affine[0, 0], *affine[0, 1])  # type: ignore


class ToCairoShape(ShapeVisitor[None]):

    def render_segment(self, seg: Segment, ctx: PyCairoContext) -> None:
        q, angle, dangle = (
            seg.q,
            tx.to_radians(seg.angle),
            tx.to_radians(seg.dangle),
        )
        end = seg.angle + seg.dangle

        for i in range(q.shape[0]):
            if tx.np.abs(dangle[i]) < 0.1:
                ctx.line_to(q[i, 0, 0], q[i, 1, 0])
            else:
                ctx.save()
                matrix = tx_to_cairo(seg.t[i : i + 1])
                ctx.transform(matrix)
                if dangle[i] < 0:
                    ctx.arc_negative(0.0, 0.0, 1.0, angle[i], end[i])
                else:
                    ctx.arc(0.0, 0.0, 1.0, angle[i], end[i])
                ctx.restore()

    def visit_path(
        self,
        path: Path,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:
        if not path.loc_trails[0].trail.closed:
            style = style.merge(Style(fill_opacity_=0))
        for loc_trail in path.loc_trails:
            p = loc_trail.location
            ctx.move_to(p[0, 0, 0], p[0, 1, 0])
            segments = loc_trail.located_segments()
            self.render_segment(segments, ctx)
            if loc_trail.trail.closed:
                ctx.close_path()

    def visit_latex(
        self,
        shape: Latex,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:
        raise NotImplementedError("Latex is not implemented")

    def visit_text(
        self,
        shape: Text,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:
        ctx.select_font_face("sans-serif")
        if shape.font_size is not None:
            ctx.set_font_size(shape.font_size)
        extents = ctx.text_extents(shape.text)

        ctx.move_to(-(extents.width / 2), (extents.height / 2))
        ctx.text_path(shape.text)

    def visit_spacer(
        self,
        shape: Spacer,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:
        return

    def visit_arrowhead(
        self,
        shape: ArrowHead,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:
        from chalk.core import get_primitives
        assert style.output_size
        scale = 0.01 * (15 / 500) * style.output_size
        render_cairo_prims(get_primitives(shape.arrow_shape.scale(scale)), ctx)

    def visit_image(
        self,
        shape: Image,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:
        surface = from_pil(shape.im)
        ctx.set_source_surface(
            surface, -(shape.width / 2), -(shape.height / 2)
        )
        ctx.paint()


def render_cairo_prims(prims: List[Primitive], ctx: PyCairoContext) -> None:
    import cairo

    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    shape_renderer = ToCairoShape()
    for prim in prims:
        for i in range(prim.transform.shape[0]):
            # apply transformation
            matrix = tx_to_cairo(prim.transform[i : i + 1])
            ctx.transform(matrix)
            prim.shape.accept(shape_renderer, ctx=ctx, style=prim.style)

            # undo transformation
            matrix.invert()
            ctx.transform(matrix)

            prim.style.render(ctx)
            ctx.stroke()


def prims_to_file(
    prims: List[Primitive], path: str, height: float, width: float
) -> None:
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(width), int(height))
    ctx = cairo.Context(surface)
    render_cairo_prims(prims, ctx)
    surface.write_to_png(path)


def render(
    self: Diagram, path: str, height: int = 128, width: Optional[int] = None
) -> None:
    """Render the diagram to a PNG file.

    Args:
        self (Diagram): Given ``Diagram`` instance.
        path (str): Path of the .png file.
        height (int, optional): Height of the rendered image.
                                Defaults to 128.
        width (Optional[int], optional): Width of the rendered image.
                                         Defaults to None.
    """

    from chalk.core import layout_primitives

    prims, height, width = layout_primitives(self, height, width)
    prims_to_file(prims, path, height, width)
