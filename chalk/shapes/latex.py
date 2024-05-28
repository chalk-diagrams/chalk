from dataclasses import dataclass
from typing import Any

from chalk.shapes.shape import Shape
from chalk.transform import P2, BoundingBox, origin
from chalk.types import Diagram
from chalk.visitor import A, ShapeVisitor


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
        eq_lines = self.eq.content.split("\n")
        c = "<g>\n" + "\n".join(eq_lines[2:-2]) + "\n</g>"

        # Undo scaling done by latextools
        # https://github.com/cduck/latextools/blob/caa15da02d88e5a4c82eb06f8fadbe48abd7ad2f/latextools/convert.py#L131
        self.width = self.eq.width * 3 / 4
        self.height = self.eq.height * 3 / 4
        self.content = c

        # From latextools Ensures no clash between multiple math statements
        id_prefix = f"embed-{hash(self.content)}-"
        self.content = (
            self.content.replace('id="', f'id="{id_prefix}')
            .replace('="url(#', f'="url(#{id_prefix}')
            .replace('xlink:href="#', f'href="#{id_prefix}')
        )

    def get_bounding_box(self) -> BoundingBox:
        eps = 1e-4
        self.bb = BoundingBox(origin, origin + P2(eps, eps))
        return self.bb

    def accept(self, visitor: ShapeVisitor[A], **kwargs: Any) -> A:
        return visitor.visit_latex(self, **kwargs)


def latex(t: str) -> Diagram:
    from chalk.core import Primitive

    return Primitive.from_shape(Latex(t))
