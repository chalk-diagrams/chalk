from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from chalk.envelope import Envelope
from chalk.trace import Trace
from chalk.transform import Affine
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


Ident = Affine.identity()
Subdiagram = Tuple[Diagram, Affine]


def get_subdiagram_envelope(
    self: Diagram, name: str, t: Affine = Ident
) -> Envelope:
    """Get the bounding envelope of the subdiagram."""
    subdiagram = self.get_subdiagram(name)
    assert subdiagram is not None, "Subdiagram does not exist"
    return subdiagram[0].get_envelope(t=subdiagram[1])  # type: ignore


def get_subdiagram_trace(self: Diagram, name: str, t: Affine = Ident) -> Trace:
    """Get the trace of the sub-diagram."""
    subdiagram = self.get_subdiagram(name)
    assert subdiagram is not None, "Subdiagram does not exist"
    return subdiagram[0].get_trace(t=subdiagram[1])  # type: ignore


class GetSubdiagram(DiagramVisitor[Optional[Subdiagram]]):
    def __init__(self, name: str, t: Affine = Ident):
        self.name = name

    def visit_primitive(
        self,
        diagram: Primitive,
        t: Affine = Ident,
    ) -> Optional[Subdiagram]:
        return None

    def visit_empty(
        self,
        diagram: Empty,
        t: Affine = Ident,
    ) -> Optional[Subdiagram]:
        return None

    def visit_compose(
        self,
        diagram: Compose,
        t: Affine = Ident,
    ) -> Optional[Subdiagram]:
        bb = diagram.diagram1.accept(self, t=t)
        if bb is None:
            bb = diagram.diagram2.accept(self, t=t)
        return bb

    def visit_apply_transform(
        self,
        diagram: ApplyTransform,
        t: Affine = Ident,
    ) -> Optional[Subdiagram]:
        return diagram.diagram.accept(self, t=t * diagram.transform)

    def visit_apply_style(
        self,
        diagram: ApplyStyle,
        t: Affine = Ident,
    ) -> Optional[Subdiagram]:
        return diagram.diagram.accept(self, t=t)

    def visit_apply_name(
        self,
        diagram: ApplyName,
        t: Affine = Ident,
    ) -> Optional[Subdiagram]:
        if self.name == diagram.dname:
            return diagram.diagram, t
        else:
            return None


def get_subdiagram(
    self: Diagram, name: str, t: Affine = Ident
) -> Optional[Subdiagram]:
    return self.accept(GetSubdiagram(name), t=t)
