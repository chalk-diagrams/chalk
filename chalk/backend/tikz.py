from __future__ import annotations

from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from chalk import transform as tx
from chalk.shape import Spacer
from chalk.style import Style
from chalk.types import Diagram
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


PyLatex = Any
PyLatexElement = Any


class ToTikZ(DiagramVisitor[List[PyLatexElement]]):
    def visit_primitive(
        self, diagram: Primitive, pylatex: PyLatex, other_style: Style
    ) -> List[PyLatexElement]:
        """Convert a diagram to SVG image."""
        transform = tx.to_tikz(diagram.transform)
        style = diagram.style.merge(other_style)
        inner = diagram.shape.render_tikz(pylatex, style)
        if not style and not transform:
            return [inner]
        else:
            options = {}
            options["cm"] = tx.to_tikz(diagram.transform)
            s = pylatex.TikZScope(options=pylatex.TikZOptions(**options))
            s.append(inner)
            return [s]

    def visit_empty(
        self, diagram: Empty, pylatex: PyLatex, style: Style
    ) -> List[PyLatexElement]:
        """Converts to SVG image."""
        return []

    def visit_compose(
        self, diagram: Compose, pylatex: PyLatex, style: Style
    ) -> List[PyLatexElement]:
        """Converts to tikz image."""
        return diagram.diagram1.accept(
            self, pylatex, style
        ) + diagram.diagram2.accept(self, pylatex, style)

    def visit_apply_transform(
        self, diagram: ApplyTransform, pylatex: PyLatex, style: Style
    ) -> List[PyLatexElement]:
        options = {}
        options["cm"] = tx.to_tikz(diagram.transform)
        s = pylatex.TikZScope(options=pylatex.TikZOptions(**options))
        for x in diagram.diagram.accept(self, pylatex, style):
            s.append(x)
        return [s]

    def visit_apply_style(
        self, diagram: ApplyStyle, pylatex: PyLatex, style: Style
    ) -> List[PyLatexElement]:
        return diagram.diagram.accept(
            self, pylatex, diagram.style.merge(style)
        )

    def visit_apply_name(
        self, diagram: ApplyName, pylatex: PyLatex, style: Style
    ) -> List[PyLatexElement]:
        return diagram.diagram.accept(self, pylatex=pylatex, style=style)


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
        for x in diagram.accept(
            ToTikZ(), pylatex, Style.root(max(height, width))
        ):
            pic.append(x)
    doc.generate_tex(path.replace(".pdf", "") + ".tex")
    doc.generate_pdf(path.replace(".pdf", ""), clean_tex=False)
