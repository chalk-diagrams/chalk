from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional

from chalk.envelope import Envelope
from chalk.segment import ray_circle_intersection
from chalk.style import Style
from chalk.trace import SignedDistance, Trace
from chalk.transform import P2, V2, BoundingBox, Ray, origin
from chalk.types import BaseElement, Drawing
from chalk.visitor import A, ShapeVisitor


@dataclass
class Shape:
    """Shape class."""

    def get_bounding_box(self) -> BoundingBox:
        raise NotImplementedError

    def get_envelope(self) -> Envelope:
        return Envelope.from_bounding_box(self.get_bounding_box())

    def get_trace(self) -> Trace:
        # default trace based on bounding box
        from chalk.path import Path

        box = self.get_bounding_box()
        return Path.rectangle(box.width, box.height).get_trace()

    def accept(self, visitor: ShapeVisitor[A], **kwargs: Any) -> A:
        raise NotImplementedError


@dataclass
class Spacer(Shape):
    """Spacer class."""

    width: float
    height: float

    def get_bounding_box(self) -> BoundingBox:
        left = origin.x - self.width / 2
        top = origin.y - self.height / 2
        tl = P2(left, top)
        br = P2(left + self.width, top + self.height)
        return BoundingBox([tl, br])

    def accept(self, visitor: ShapeVisitor[A], **kwargs: Any) -> A:
        return visitor.visit_spacer(self, **kwargs)
