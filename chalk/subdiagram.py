from typing import Optional, Tuple

from chalk.envelope import Envelope
from chalk.trace import Trace
from chalk.transform import Affine
from chalk.types import Diagram

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


class GetSubdiagram:
    def visit_primitive(
        self, diagram, name: str, t: Affine = Ident
    ) -> Optional[Tuple[Diagram, Affine]]:
        return None

    def visit_empty(
        self, diagram, name: str, t: Affine = Ident
    ) -> Optional[Tuple[Diagram, Affine]]:
        return None

    def visit_compose(
        self, diagram, name: str, t: Affine = Ident
    ) -> Optional[Tuple[Diagram, Affine]]:
        """Get the bounding envelope of the sub-diagram."""
        bb = diagram.diagram1.accept(self, name, t)
        if bb is None:
            bb = diagram.diagram2.accept(self, name, t)
        return bb

    def visit_apply_transform(
        self, diagram, name: str, t: Affine = Ident
    ) -> Optional[Tuple[Diagram, Affine]]:
        """Get the bounding envelope of the sub-diagram."""
        return diagram.diagram.accept(self, name, t * diagram.transform)

    def visit_apply_style(
        self, diagram, name: str, t: Affine = Ident
    ) -> Optional[Tuple[Diagram, Affine]]:
        return diagram.diagram.accept(self, name, t)

    def visit_apply_name(
        self, diagram, name: str, t: Affine = Ident
    ) -> Optional[Tuple[Diagram, Affine]]:
        """Get the bounding envelope of the sub-diagram."""
        if name == diagram.dname:
            return diagram.diagram, t
        else:
            return None
