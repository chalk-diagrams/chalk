from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from chalk import transform as tx
from chalk.shape import Spacer
from chalk.style import Style
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


PyLatex = Any
PyLatexElement = Any

EMPTY_STYLE = Style.empty()


class ToTikZ(DiagramVisitor[List[PyLatexElement]]):
    def __init__(self, pylatex: PyLatex):
        self.pylatex = pylatex

    def visit_primitive(
        self, diagram: Primitive, style: Style = EMPTY_STYLE
    ) -> List[PyLatexElement]:
        transform = tx.to_tikz(diagram.transform)
        style_new = diagram.style.merge(style)
        inner = diagram.shape.render_tikz(self.pylatex, style_new)
        if not style_new and not transform:
            return [inner]
        else:
            options = {"cm": tx.to_tikz(diagram.transform)}
            s = self.pylatex.TikZScope(
                options=self.pylatex.TikZOptions(**options)
            )
            s.append(inner)
            return [s]

    def visit_empty(
        self, diagram: Empty, style: Style = EMPTY_STYLE
    ) -> List[PyLatexElement]:
        return []

    def visit_compose(
        self, diagram: Compose, style: Style = EMPTY_STYLE
    ) -> List[PyLatexElement]:
        elems1 = diagram.diagram1.accept(self, style=style)
        elems2 = diagram.diagram2.accept(self, style=style)
        return elems1 + elems2

    def visit_apply_transform(
        self, diagram: ApplyTransform, style: Style = EMPTY_STYLE
    ) -> List[PyLatexElement]:
        options = {"cm": tx.to_tikz(diagram.transform)}
        s = self.pylatex.TikZScope(options=self.pylatex.TikZOptions(**options))
        for x in diagram.diagram.accept(self, style=style):
            s.append(x)
        return [s]

    def visit_apply_style(
        self, diagram: ApplyStyle, style: Style = EMPTY_STYLE
    ) -> List[PyLatexElement]:
        style_new = diagram.style.merge(style)
        return diagram.diagram.accept(self, style=style_new)

    def visit_apply_name(
        self, diagram: ApplyName, style: Style = EMPTY_STYLE
    ) -> List[PyLatexElement]:
        return diagram.diagram.accept(self, style=style)


def to_tikz(
    self: Diagram, pylatex: PyLatex, style: Style
) -> List[PyLatexElement]:
    return self.accept(ToTikZ(pylatex), style=style)


def render(self: Diagram, path: str, height: int = 128) -> None:
    # Hack: Convert roughly from px to pt. Assume 300 dpi.
    heightpt = height / 4.3
    try:
        import pylatex
    except ImportError:
        print("Render PDF requires pylatex installation.")
        return

    pad = 0.05
    envelope = self.get_envelope()
    assert envelope is not None

    # infer width to preserve aspect ratio
    width = heightpt * (envelope.width / envelope.height)
    # determine scale to fit the largest axis in the target frame size
    if envelope.width - width <= envelope.height - heightpt:
        α = heightpt / ((1 + pad) * envelope.height)
    else:
        α = width / ((1 + pad) * envelope.width)
    x, _ = pad * heightpt, pad * width

    # create document
    doc = pylatex.Document(documentclass="standalone")
    # document_options= pylatex.TikZOptions(margin=f"{{{x}pt {x}pt {y}pt {y}pt}}"))
    # add our sample drawings
    diagram = self.scale(α).reflect_y().pad(1 + pad)
    envelope = diagram.get_envelope()
    assert envelope is not None
    from chalk.core import Primitive

    padding = Primitive.from_shape(
        Spacer(envelope.width, envelope.height)
    ).translate(envelope.center.x, envelope.center.y)
    diagram = diagram + padding
    with doc.create(pylatex.TikZ()) as pic:
        for x in to_tikz(diagram, pylatex, Style.root(max(height, width))):
            pic.append(x)
    doc.generate_tex(path.replace(".pdf", "") + ".tex")
    doc.generate_pdf(path.replace(".pdf", ""), clean_tex=False)
