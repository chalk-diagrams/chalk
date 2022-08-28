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
        c = "<g>\n" + "\n".join(self.eq.content.split("\n")[2:-2]) + "\n</g>"
        self.width = self.eq.width
        self.height = self.eq.height
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
        self.bb = BoundingBox([origin, origin + P2(eps, eps)])
        return self.bb

    def accept(self, visitor: ShapeVisitor[A], **kwargs: Any) -> A:
        return visitor.visit_latex(self, **kwargs)


def latex(t: str) -> Diagram:
    from chalk.core import Primitive

    return Primitive.from_shape(Latex(t))
