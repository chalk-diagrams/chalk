from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from chalk import transform as tx
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
)
from chalk.style import Style
from chalk.transform import P2, origin
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


PyLatex = Any
PyLatexElement = Any

EMPTY_STYLE = Style.empty()


def tx_to_tikz(affine: tx.Affine) -> str:
    def convert(
        a: float, b: float, c: float, d: float, e: float, f: float
    ) -> str:
        return f"{{{a}, {d}, {b}, {e}, ({c}, {f})}}"

    return convert(*affine[:6])


class ToTikZ(DiagramVisitor[List[PyLatexElement]]):
    def __init__(self, pylatex: PyLatex):
        self.pylatex = pylatex
        self.shape_renderer = ToTikZShape(pylatex)

    def visit_primitive(
        self, diagram: Primitive, style: Style = EMPTY_STYLE
    ) -> List[PyLatexElement]:
        transform = tx_to_tikz(diagram.transform)
        style_new = diagram.style.merge(style)
        inner = diagram.shape.accept(self.shape_renderer, style=style_new)
        if not style_new and not transform:
            return [inner]
        else:
            options = {"cm": tx_to_tikz(diagram.transform)}
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
        options = {"cm": tx_to_tikz(diagram.transform)}
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


class ToTikZShape(ShapeVisitor[PyLatexElement]):
    def __init__(self, pylatex: PyLatex):
        self.pylatex = pylatex

    def render_segment(
        self, pts: PyLatexElement, seg: SegmentLike, p: P2
    ) -> None:
        q = seg.q + p
        if isinstance(seg, Segment):
            pts.append("--")
            pts.append(self.pylatex.TikZCoordinate(q.x, q.y))
        elif isinstance(seg, ArcSegment):
            start = (-seg.center).angle
            end = (seg.q - seg.center).angle
            det: float = seg.t.determinant  # type: ignore
            if det * seg.dangle < 0 and end > start:
                end = end - 360
            if det * seg.dangle > 0 and end < start:
                end = end + 360
            end_ang = end - seg.rot
            pts._arg_list.append(
                self.pylatex.TikZUserPath(
                    f"""{{[rotate={seg.rot}] arc [
                           start angle={start-seg.rot}, end angle={end_ang},
                           x radius={seg.r_x}, y radius={seg.r_y}]}}"""
                )
            )

    def visit_path(
        self, path: Path, style: Style = EMPTY_STYLE
    ) -> PyLatexElement:
        pts = self.pylatex.TikZPathList()
        if not path.loc_trails[0].trail.closed:
            style = style.fill_opacity(0)

        for loc_trail in path.loc_trails:
            for i, (seg, p) in enumerate(loc_trail.located_segments()):
                if i == 0:
                    pts.append(self.pylatex.TikZCoordinate(p.x, p.y))
                self.render_segment(pts, seg, p)
            if loc_trail.trail.closed:
                pts.append("--")
                pts._arg_list.append(self.pylatex.TikZUserPath("cycle"))
        return self.pylatex.TikZDraw(
            pts,
            options=self.pylatex.TikZOptions(**style.to_tikz(self.pylatex)),
        )

    def visit_latex(self, shape: Latex) -> PyLatexElement:
        raise NotImplementedError("Latex is not implemented")

    def visit_text(
        self, shape: Text, style: Style = EMPTY_STYLE
    ) -> PyLatexElement:
        opts = {}
        opts["font"] = "\\small\\sffamily"
        opts["scale"] = str(
            3.5 * (1 if shape.font_size is None else shape.font_size)
        )
        styles = style.to_tikz(self.pylatex)
        if styles["fill"] is not None:
            opts["text"] = styles["fill"]
        return self.pylatex.TikZNode(
            text=shape.text,
            # Scale parameters based on observations
            options=self.pylatex.TikZOptions(**opts),
        )

    def visit_spacer(
        self, shape: Spacer, style: Style = EMPTY_STYLE
    ) -> PyLatexElement:
        left = origin.x - shape.width / 2
        top = origin.y - shape.height / 2
        return self.pylatex.TikZPath(
            [
                self.pylatex.TikZCoordinate(left, top),
                "rectangle",
                self.pylatex.TikZCoordinate(
                    left + shape.width, top + shape.height
                ),
            ]
        )

    def visit_arrowhead(
        self, shape: ArrowHead, style: Style = EMPTY_STYLE
    ) -> PyLatexElement:
        assert style.output_size
        scale = 0.01 * 3 * (15 / 500) * style.output_size
        s = self.pylatex.TikZScope()
        for inner in to_tikz(
            shape.arrow_shape.scale(scale), self.pylatex, style
        ):
            s.append(inner)
        return s

    def visit_image(
        self, shape: Image, style: Style = EMPTY_STYLE
    ) -> PyLatexElement:
        assert False, "No tikz image renderer"


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
