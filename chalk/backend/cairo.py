from __future__ import annotations

from typing import Any, Optional, List, TYPE_CHECKING

from chalk import transform as tx
from chalk.style import Style
from chalk.types import Diagram
from chalk.transform import Affine, unit_x, unit_y
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import (
        Primitive,
        Empty,
        Compose,
        ApplyTransform,
        ApplyStyle,
        ApplyName,
    )


Ident = Affine.identity()
PyCairoContext = Any


class ToList(DiagramVisitor[List[Primitive]]):
    """Compiles a `Diagram` to a list of `Primitive`s. The transfomation `t`
    is accumulated upwards, from the tree's leaves.
    """

    def visit_primitive(
        self, diagram: Primitive, t: Affine = Ident, **kwargs: Any
    ) -> List[Primitive]:
        return [diagram.apply_transform(t)]

    def visit_empty(
        self, diagram: Empty, t: Affine = Ident, **kwargs: Any
    ) -> List[Primitive]:
        return []

    def visit_compose(
        self, diagram: Compose, t: Affine = Ident, **kwargs: Any
    ) -> List[Primitive]:
        return diagram.diagram1.accept(self, t=t) + diagram.diagram2.accept(
            self, t=t
        )

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine = Ident, **kwargs: Any
    ) -> List[Primitive]:
        t_new = t * diagram.transform
        return [
            prim.apply_transform(t_new)
            for prim in diagram.diagram.accept(self, t=t)
        ]

    def visit_apply_style(
        self, diagram: ApplyStyle, t: Affine = Ident, **kwargs: Any
    ) -> List[Primitive]:
        return [
            prim.apply_style(diagram.style)
            for prim in diagram.diagram.accept(self, t=t)
        ]

    def visit_apply_name(
        self, diagram: ApplyName, t: Affine = Ident, **kwargs: Any
    ) -> List[Primitive]:
        return [prim for prim in diagram.diagram.accept(self, t=t)]


def render_cairo_prims(
    base: Diagram, ctx: PyCairoContext, style: Style
) -> None:
    base = base._style(style)
    for prim in base.accept(ToList()):
        # apply transformation
        matrix = tx.to_cairo(prim.transform)
        ctx.transform(matrix)

        prim.shape.render(ctx, prim.style)

        # undo transformation
        matrix.invert()
        ctx.transform(matrix)
        prim.style.render(ctx)
        ctx.stroke()

def render(
    self, path: str, height: int = 128, width: Optional[int] = None
) -> None:
    """Render the diagram to a PNG file.

    Args:
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
