from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from chalk.core import (
        ApplyName,
        ApplyStyle,
        ApplyTransform,
        Compose,
        Empty,
        Primitive,
    )


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
