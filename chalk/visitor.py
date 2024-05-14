from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from chalk.ArrowHead import ArrowHead
    from chalk.core import (
        ApplyName,
        ApplyStyle,
        ApplyTransform,
        Compose,
        Empty,
        Primitive,
    )
    from chalk.Path import Path
    from chalk.shapes import Image, Latex, Spacer, Text
    from chalk.types import Monoid

    A = TypeVar("A", bound=Monoid)
else:
    A = TypeVar("A")

B = TypeVar("B")


class DiagramVisitor(Generic[A, B]):
    A_type: type[A]

    def visit_primitive(self, diagram: Primitive, arg: B) -> A:
        "Primitive detauls to empty"
        return self.A_type.empty()

    def visit_empty(self, diagram: Empty, arg: B) -> A:
        "Empty defaults to empty"
        return self.A_type.empty()

    def visit_compose(self, diagram: Compose, arg: B) -> A:
        "Compose defaults to monoid over children"
        return self.A_type.concat(
            [d.accept(self, arg) for d in diagram.diagrams]
        )

    def visit_apply_transform(self, diagram: ApplyTransform, arg: B) -> A:
        "Defaults to pass over"
        return diagram.diagram.accept(self, arg)

    def visit_apply_style(self, diagram: ApplyStyle, arg: B) -> A:
        "Defaults to pass over"
        return diagram.diagram.accept(self, arg)

    def visit_apply_name(self, diagram: ApplyName, arg: B) -> A:
        "Defaults to pass over"
        return diagram.diagram.accept(self, arg)


C = TypeVar("C")


class ShapeVisitor(Generic[C]):
    def visit_path(self, shape: Path) -> C:
        raise NotImplementedError

    def visit_latex(self, shape: Latex) -> C:
        raise NotImplementedError

    def visit_text(self, shape: Text) -> C:
        raise NotImplementedError

    def visit_spacer(self, shape: Spacer) -> C:
        raise NotImplementedError

    def visit_arrowhead(self, shape: ArrowHead) -> C:
        raise NotImplementedError

    def visit_image(self, shape: Image) -> C:
        raise NotImplementedError
