from dataclasses import dataclass
from typing import Any, Optional

import chalk.transform as tx
from chalk.shapes.shape import Shape
from chalk.transform import P2, BoundingBox
from chalk.types import Diagram
from chalk.visitor import C, ShapeVisitor


@dataclass(unsafe_hash=True, frozen=True)
class Text(Shape):
    """Text class."""

    text: str
    font_size: Optional[float]

    def get_bounding_box(self) -> BoundingBox:
        # Text doesn't have a bounding box since we can't accurately know
        # its size for all backends.
        eps = 1e-4
        self.bb = BoundingBox(tx.X.origin, tx.X.origin + P2(eps, eps))
        return self.bb

    def accept(self, visitor: ShapeVisitor[C], **kwargs: Any) -> C:
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
