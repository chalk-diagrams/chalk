from dataclasses import dataclass
from typing import Any, Optional

from chalk.shapes.shape import Shape
from chalk.transform import P2, BoundingBox, origin
from chalk.types import Diagram
from chalk.visitor import A, ShapeVisitor


@dataclass
class Text(Shape):
    """Text class."""

    text: str
    font_size: Optional[float]

    def get_bounding_box(self) -> BoundingBox:
        # Text doesn't have a bounding box since we can't accurately know
        # its size for all backends.
        eps = 1e-4
        self.bb = BoundingBox([origin, origin + P2(eps, eps)])
        return self.bb

    def accept(self, visitor: ShapeVisitor[A], **kwargs: Any) -> A:
        return visitor.visit_text(self, **kwargs)


def text(t: str, size: Optional[float]) -> Diagram:
    """
    Draw some text.

    Args:
       t (str): The text string.
       size (Optional[float]): Size of the text.

    Returns:
       Diagram

    """
    from chalk.core import Primitive

    return Primitive.from_shape(Text(t, font_size=size))
