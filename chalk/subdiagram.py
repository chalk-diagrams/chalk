from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, List, Tuple, Union

from chalk.envelope import Envelope
from chalk.trace import Trace
from chalk.transform import Affine, P2, V2, apply_affine, origin
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
AtomicName = Any
Name = Tuple[AtomicName, ...]


@dataclass
class Subdiagram:
    diagram: Diagram
    transform: Affine
    # style: Style

    def get_location(self) -> P2:
        return apply_affine(self.transform, origin)

    def get_envelope(self) -> Envelope:
        return self.diagram.get_envelope().apply_transform(self.transform)

    def get_trace(self) -> Trace:
        return self.diagram.get_trace().apply_transform(self.transform)

    def boundary_from(self, v: V2) -> P2:
        """Returns the furthest point on the boundary of the subdiagram,
        starting from the local origin of the subdigram and going in the
        direction of the given vector `v`.
        """
        o = self.get_location()
        p = self.get_trace().trace_p(o, -v)
        if not p:
            return origin
        else:
            return p


class GetSubdiagram(DiagramVisitor[Optional[Subdiagram]]):
    def __init__(self, name: Name, t: Affine = Ident):
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
            return Subdiagram(diagram.diagram, t)
        else:
            return None


def get_subdiagram(self: Diagram, *name: AtomicName) -> Optional[Subdiagram]:
    return self.accept(GetSubdiagram(name), t=Ident)


def with_names(
    self: Diagram,
    names: List[Name],
    f: Callable[[List[Subdiagram], Diagram], Diagram],
) -> Diagram:
    # NOTE Instead of performing a pass of the AST for each `name` in `names`,
    # it might be more efficient to retrieve all named subdiagrams using the
    # `get_sub_map` function and then filter the subdiagrams specified by
    # `names`.
    subs = [self.get_subdiagram(*name) for name in names]
    if any(sub is None for sub in subs):
        return self
    else:
        return f(subs, self)


SubMap = Dict[Name, List[Subdiagram]]


class GetSubMap(DiagramVisitor[SubMap]):
    def visit_primitive(
        self,
        diagram: Primitive,
        t: Affine = Ident,
    ) -> SubMap:
        return {}

    def visit_empty(
        self,
        diagram: Empty,
        t: Affine = Ident,
    ) -> SubMap:
        return {}

    def visit_compose(
        self,
        diagram: Compose,
        t: Affine = Ident,
    ) -> SubMap:
        d1 = diagram.diagram1.accept(self, t=t)
        d2 = diagram.diagram2.accept(self, t=t)
        return self._union(d1, d2)

    def visit_apply_transform(
        self,
        diagram: ApplyTransform,
        t: Affine = Ident,
    ) -> SubMap:
        return diagram.diagram.accept(self, t=t * diagram.transform)

    def visit_apply_style(
        self,
        diagram: ApplyStyle,
        t: Affine = Ident,
    ) -> SubMap:
        return diagram.diagram.accept(self, t=t)

    def visit_apply_name(
        self,
        diagram: ApplyName,
        t: Affine = Ident,
    ) -> SubMap:
        d1 = {tuple(diagram.dname): [Subdiagram(diagram.diagram, t)]}
        d2 = diagram.diagram.accept(self, t=t)
        return self._union(d1, d2)

    @staticmethod
    def _union(d1: SubMap, d2: SubMap) -> SubMap:
        return {k: d1.get(k, []) + d2.get(k, []) for k in set(d1) | set(d2)}


def get_sub_map(self: Diagram, t: Affine = Ident) -> SubMap:
    """Retrieves all named subdiagrams in the given diagram and accumulates
    them in a dictionary (map) indexed by their name.
    """
    return self.accept(GetSubMap(), t=t)
