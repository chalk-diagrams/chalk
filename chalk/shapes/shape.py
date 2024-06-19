from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chalk.transform as tx
# from chalk.envelope import Envelope
from chalk.trace import Trace
from chalk.trail import Trail
from chalk.transform import BoundingBox
from chalk.types import Diagram
from chalk.visitor import C, ShapeVisitor


@dataclass
class Shape:
    """Shape class."""

    def get_bounding_box(self) -> BoundingBox:
        raise NotImplementedError

    # def get_envelope(self) -> Envelope:
    #     return Envelope.from_bounding_box(self.get_bounding_box())

    def get_trace(self) -> Trace:
        box = self.get_bounding_box()
        return (
            Trail.rectangle(box.width, box.height)
            .stroke()
            .center_xy()
            .get_trace()
        )

    def accept(self, visitor: ShapeVisitor[C], **kwargs: Any) -> C:
        raise NotImplementedError

    def stroke(self) -> Diagram:
        """Returns a primitive (shape) with strokes

        Returns:
            Diagram: A diagram.
        """
        from chalk.core import Primitive

        return Primitive.from_shape(self)


@dataclass
class Spacer(Shape):
    """Spacer class."""

    width: tx.Scalars
    height: tx.Scalars

    def get_bounding_box(self) -> BoundingBox:
        left = -self.width / 2
        top = -self.height / 2
        tl = tx.P2(left, top)
        br = tx.P2(left + self.width, top + self.height)
        return BoundingBox(tl, br)

    def accept(self, visitor: ShapeVisitor[C], **kwargs: Any) -> C:
        return visitor.visit_spacer(self, **kwargs)
