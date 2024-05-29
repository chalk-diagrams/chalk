from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from chalk.monoid import MList
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
from chalk.style import Style
from chalk.transform import P2_t, Affine
import chalk.transform as tx
from chalk.types import Diagram
from chalk.visitor import DiagramVisitor, ShapeVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyName, ApplyStyle, ApplyTransform, Primitive


Ident = tx.ident
PyCairoContext = Any
EMPTY_STYLE = Style.empty()


def tx_to_cairo(affine: Affine) -> Any:
    import cairo

    def convert(a, b, c, d, e, f):  # type: ignore
        return cairo.Matrix(a, d, b, e, c, f)  # type: ignore

    return convert(*affine[0, 0], *affine[0, 1])  # type: ignore


class ToList(DiagramVisitor[MList[Any], Affine]):
    """Compiles a `Diagram` to a list of `Primitive`s. The transformation `t`
    is accumulated upwards, from the tree's leaves.
    """

    A_type = MList[Any]

    def visit_primitive(
        self, diagram: Primitive, t: Affine = Ident
    ) -> MList[Primitive]:
        return MList([diagram.apply_transform(t)])

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine = Ident
    ) -> MList[Primitive]:
        t_new = t @ diagram.transform
        return MList(
            [
                prim.apply_transform(t_new)
                for prim in diagram.diagram.accept(self, t).data
            ]
        )

    def visit_apply_style(
        self, diagram: ApplyStyle, t: Affine = Ident
    ) -> MList[Primitive]:
        return MList(
            [
                prim.apply_style(diagram.style)
                for prim in diagram.diagram.accept(self, t).data
            ]
        )

    def visit_apply_name(
        self, diagram: ApplyName, t: Affine = Ident
    ) -> MList[Primitive]:
        return MList([prim for prim in diagram.diagram.accept(self, t).data])


class ToCairoShape(ShapeVisitor[None]):

    def render_segment(
        self, seg: Segment, ctx: PyCairoContext
    ) -> None:
        q, angle, dangle = seg.q, tx.to_radians(seg.angle), tx.to_radians(seg.dangle)
        end = seg.angle + seg.dangle

        for i in range(q.shape[0]):
            ctx.save()
            matrix = tx_to_cairo(seg.t[i][None])
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
        style: Style = EMPTY_STYLE,
    ) -> None:
        if not path.loc_trails[0].trail.closed:
            style.fill_opacity_ = 0
        for loc_trail in path.loc_trails:
            p = loc_trail.location
            ctx.move_to(p[0, 0, 0], p[0, 1,0])
            segments = loc_trail.located_segments()
            self.render_segment(segments, ctx)
            if loc_trail.trail.closed:
                ctx.close_path()

    def visit_latex(
        self,
        shape: Latex,
        ctx: PyCairoContext = None,
        style: Style = EMPTY_STYLE,
    ) -> None:
        raise NotImplementedError("Latex is not implemented")

    def visit_text(
        self,
        shape: Text,
        ctx: PyCairoContext = None,
        style: Style = EMPTY_STYLE,
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
        style: Style = EMPTY_STYLE,
    ) -> None:
        return

    def visit_arrowhead(
        self,
        shape: ArrowHead,
        ctx: PyCairoContext = None,
        style: Style = EMPTY_STYLE,
    ) -> None:
        assert style.output_size
        scale = 0.01 * (15 / 500) * style.output_size
        render_cairo_prims(shape.arrow_shape.scale(scale), ctx, style)

    def visit_image(
        self,
        shape: Image,
        ctx: PyCairoContext = None,
        style: Style = EMPTY_STYLE,
    ) -> None:
        surface = from_pil(shape.im)
        ctx.set_source_surface(
            surface, -(shape.width / 2), -(shape.height / 2)
        )
        ctx.paint()


def render_cairo_prims(
    base: Diagram, ctx: PyCairoContext, style: Style
) -> None:
    base = base._style(style)
    shape_renderer = ToCairoShape()
    for prim in base.accept(ToList(), Ident):
        # apply transformation
        for i in range(prim.transform.shape[0]):
            matrix = tx_to_cairo(prim.transform[i:i+1])
            ctx.transform(matrix)
            prim.shape.accept(shape_renderer, ctx=ctx, style=prim.style)

            # undo transformation
            matrix.invert()
            ctx.transform(matrix)

            prim.style.render(ctx)
            ctx.stroke()


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
    import cairo

    envelope = self.get_envelope()
    assert envelope is not None

    pad = 0.05

    # infer width to preserve aspect ratio
    width = width or int(height * envelope.width / envelope.height)

    # determine scale to fit the largest axis in the target frame size
    if envelope.width - width <= envelope.height - height:
        α = height / ((1 + pad) * envelope.height)
    else:
        α = width / ((1 + pad) * envelope.width)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    s = self.scale(α).center_xy().pad(1 + pad)
    e = s.get_envelope()
    assert e is not None
    s = s.translate(e(-tx.unit_x), e(-tx.unit_y))
    render_cairo_prims(s, ctx, Style.root(max(width, height)))
    surface.write_to_png(path)
