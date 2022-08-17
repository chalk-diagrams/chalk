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

A = TypeVar("A")


class DiagramVisitor(Generic[A]):
    def visit_primitive(self, diagram: Primitive) -> A:
        raise NotImplementedError

    def visit_empty(self, diagram: Empty) -> A:
        raise NotImplementedError

    def visit_compose(self, diagram: Compose) -> A:
        raise NotImplementedError

    def visit_apply_transform(self, diagram: ApplyTransform) -> A:
        raise NotImplementedError

    def visit_apply_style(self, diagram: ApplyStyle) -> A:
        raise NotImplementedError

    def visit_apply_name(self, diagram: ApplyName) -> A:
        raise NotImplementedError


class ShapeVisitor(Generic[A]):
    def visit_path(self, shape: Path) -> A:
        raise NotImplementedError

    def visit_latex(self, shape: Latex) -> A:
        raise NotImplementedError

    def visit_text(self, shape: Text) -> A:
        raise NotImplementedError

    def visit_spacer(self, shape: Spacer) -> A:
        raise NotImplementedError

    def visit_arrowhead(self, shape: ArrowHead) -> A:
        raise NotImplementedError

    def visit_image(self, shape: Image) -> A:
        raise NotImplementedError
