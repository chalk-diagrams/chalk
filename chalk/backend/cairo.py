from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

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
    from_pil,
)
from chalk.style import Style
from chalk.transform import P2, Affine, to_radians, unit_x, unit_y
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


Ident = Affine.identity()
PyCairoContext = Any
EMPTY_STYLE = Style.empty()


def tx_to_cairo(affine: Affine) -> Any:
    import cairo

    def convert(a, b, c, d, e, f):  # type: ignore
        return cairo.Matrix(a, d, b, e, c, f)  # type: ignore

    return convert(*affine[:6])  # type: ignore


class ToList(DiagramVisitor[List["Primitive"]]):
    """Compiles a `Diagram` to a list of `Primitive`s. The transfomation `t`
    is accumulated upwards, from the tree's leaves.
    """

    def visit_primitive(
        self, diagram: Primitive, t: Affine = Ident
    ) -> List[Primitive]:
        return [diagram.apply_transform(t)]

    def visit_empty(
        self, diagram: Empty, t: Affine = Ident
    ) -> List[Primitive]:
        return []

    def visit_compose(
        self, diagram: Compose, t: Affine = Ident
    ) -> List[Primitive]:
        elems1 = diagram.diagram1.accept(self, t=t)
        elems2 = diagram.diagram2.accept(self, t=t)
        return elems1 + elems2

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine = Ident
    ) -> List[Primitive]:
        t_new = t * diagram.transform
        return [
            prim.apply_transform(t_new)
            for prim in diagram.diagram.accept(self, t=t)
        ]

    def visit_apply_style(
        self, diagram: ApplyStyle, t: Affine = Ident
    ) -> List[Primitive]:
        return [
            prim.apply_style(diagram.style)
            for prim in diagram.diagram.accept(self, t=t)
        ]

    def visit_apply_name(
        self, diagram: ApplyName, t: Affine = Ident
    ) -> List[Primitive]:
        return [prim for prim in diagram.diagram.accept(self, t=t)]


class ToCairoShape(ShapeVisitor[None]):
    def render_segment(
        self, seg: SegmentLike, ctx: PyCairoContext, p: P2
    ) -> None:
        q = seg.q + p
        if isinstance(seg, Segment):
            ctx.line_to(q.x, q.y)
        elif isinstance(seg, ArcSegment):
            end = seg.angle + seg.dangle
            ctx.save()
            matrix = tx_to_cairo(Affine.translation(p) * seg.t)
            ctx.transform(matrix)
            if seg.dangle < 0:
                ctx.arc_negative(
                    0.0, 0.0, 1.0, to_radians(seg.angle), to_radians(end)
                )
            else:
                ctx.arc(0.0, 0.0, 1.0, to_radians(seg.angle), to_radians(end))
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
            for i, (seg, p) in enumerate(loc_trail.located_segments()):
                if i == 0:
                    ctx.move_to(p.x, p.y)
                self.render_segment(seg, ctx, p)
            if loc_trail.trail.closed:
                ctx.close_path()

    def visit_latex(self, shape: Latex) -> None:
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
    for prim in base.accept(ToList()):
        # apply transformation
        matrix = tx_to_cairo(prim.transform)
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
    s = s.translate(e(-unit_x), e(-unit_y))
    render_cairo_prims(s, ctx, Style.root(max(width, height)))
    surface.write_to_png(path)
