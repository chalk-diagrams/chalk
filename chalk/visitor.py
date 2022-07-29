from __future__ import annotations

from typing import Any, Generic, TypeVar, TYPE_CHECKING


if TYPE_CHECKING:
    from chalk.core import Primitive, Empty, Compose, ApplyTransform, ApplyStyle, ApplyName


A = TypeVar("A")


class DiagramVisitor(Generic[A]):
    def visit_primitive(self, diagram: "Primitive", **kwargs: Any) -> A:
        raise NotImplementedError

    def visit_empty(self, diagram: "Empty", **kwargs: Any) -> A:
        raise NotImplementedError

    def visit_compose(self, diagram: "Compose", **kwargs: Any) -> A:
        raise NotImplementedError

    def visit_apply_transform(self, diagram: "ApplyTransform", **kwargs: Any) -> A:
        raise NotImplementedError

    def visit_apply_style(self, diagram: "ApplyStyle", **kwargs: Any) -> A:
        raise NotImplementedError

    def visit_apply_name(self, diagram: "ApplyName", **kwargs: Any) -> A:
        raise NotImplementedError
