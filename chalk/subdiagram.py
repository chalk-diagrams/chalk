from typing import Optional, Tuple, TYPE_CHECKING

from chalk.envelope import Envelope
from chalk.trace import Trace
from chalk.transform import Affine
from chalk.types import Diagram
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import Primitive, Empty, Compose, ApplyTransform, ApplyStyle, ApplyName


Ident = Affine.identity()
Subdiagram = Tuple[Diagram, Affine]


def get_subdiagram_envelope(
    self, name: str, t: Affine = Ident
) -> Envelope:
    """Get the bounding envelope of the sub-diagram."""
    subdiagram = self.get_subdiagram(name)
    assert subdiagram is not None, "Subdiagram does not exist"
    return subdiagram[0].get_envelope(subdiagram[1])


def get_subdiagram_trace(self, name: str, t: Affine = Ident) -> Trace:
    """Get the trace of the sub-diagram."""
    subdiagram = self.get_subdiagram(name)
    assert subdiagram is not None, "Subdiagram does not exist"
    return subdiagram[0].get_trace(subdiagram[1])


class GetSubdiagram(DiagramVisitor[Subdiagram]):
    def visit_primitive(
            self, diagram: "Primitive", name: str, t: Affine = Ident
    ) -> Optional[Subdiagram]:
        return None

    def visit_empty(
            self, diagram: "Empty", name: str, t: Affine = Ident
    ) -> Optional[Subdiagram]:
        return None

    def visit_compose(
            self, diagram: "Compose", name: str, t: Affine = Ident
    ) -> Optional[Subdiagram]:
        """Get the bounding envelope of the sub-diagram."""
        bb = diagram.diagram1.accept(self, name, t)
        if bb is None:
            bb = diagram.diagram2.accept(self, name, t)
        return bb

    def visit_apply_transform(
            self, diagram: "ApplyTransform", name: str, t: Affine = Ident
    ) -> Optional[Subdiagram]:
        """Get the bounding envelope of the sub-diagram."""
        return diagram.diagram.accept(self, name, t * diagram.transform)

    def visit_apply_style(
            self, diagram: "ApplyStyle", name: str, t: Affine = Ident
    ) -> Optional[Subdiagram]:
        return diagram.diagram.accept(self, name, t)

    def visit_apply_name(
            self, diagram: "ApplyName", name: str, t: Affine = Ident
    ) -> Optional[Subdiagram]:
        """Get the bounding envelope of the sub-diagram."""
        if name == diagram.dname:
            return diagram.diagram, t
        else:
            return None
