from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import chalk.transform as tx
from chalk.envelope import Envelope
from chalk.monoid import Maybe, Monoid
from chalk.trace import Trace
from chalk.transform import Affine, P2_t, V2_t
from chalk.types import Diagram
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyName, ApplyTransform, Compose


AtomicName = Any


@dataclass
class Name:
    atomic_names: Tuple[AtomicName, ...]

    def __init__(self, atomic_name: AtomicName):
        self.atomic_names = (atomic_name,)

    def __hash__(self) -> int:
        return hash(self.atomic_names)

    def __str__(self) -> str:
        return "·".join(map(str, self.atomic_names))

    def __add__(self, other: Name) -> Name:
        new_name = Name(None)
        new_name.atomic_names = self.atomic_names + other.atomic_names
        return new_name

    def qualify(self, name: Name) -> Name:
        return name + self


@dataclass
class Subdiagram(Monoid):
    diagram: Diagram
    transform: Affine
    # style: Style

    def get_location(self) -> P2_t:
        return self.transform @ tx.origin

    def get_envelope(self) -> Envelope:
        return self.diagram.get_envelope().apply_transform(self.transform)

    def get_trace(self) -> Trace:
        return self.diagram.get_trace().apply_transform(self.transform)

    def boundary_from(self, v: V2_t) -> P2_t:
        """Returns the furthest point on the boundary of the subdiagram,
        starting from the local origin of the subdiagram and going in the
        direction of the given vector `v`.
        """
        o = self.get_location()
        p, m = self.get_trace().trace_p(o, -v)
        return tx.np.where(m, p, tx.origin)


class GetSubdiagram(DiagramVisitor[Maybe[Subdiagram], Affine]):
    A_type = Maybe[Subdiagram]

    def __init__(self, name: Name, t: Affine):
        self.name = name

    def visit_compose(
        self,
        diagram: Compose,
        t: Affine
    ) -> Maybe[Subdiagram]:
        for d in diagram.diagrams:
            bb = d.accept(self, t)
            if bb.data is not None:
                return bb
        return Maybe.empty()

    def visit_apply_transform(
        self,
        diagram: ApplyTransform,
        t: Affine
    ) -> Maybe[Subdiagram]:
        return diagram.diagram.accept(self, t @ diagram.transform)

    def visit_apply_name(
        self,
        diagram: ApplyName,
        t: Affine
    ) -> Maybe[Subdiagram]:
        if self.name == diagram.dname:
            return Maybe(Subdiagram(diagram.diagram, t))
        else:
            return diagram.diagram.accept(self, t)


def get_subdiagram(self: Diagram, name: Name) -> Optional[Subdiagram]:
    return self.accept(GetSubdiagram(name), tx.ident).data


def with_names(
    self: Diagram,
    names: List[Name],
    f: Callable[[List[Subdiagram], Diagram], Diagram],
) -> Diagram:
    # NOTE Instead of performing a pass of the AST for each `name` in `names`,
    # it might be more efficient to retrieve all named subdiagrams using the
    # `get_sub_map` function and then filter the subdiagrams specified by
    # `names`.
    subs = [self.get_subdiagram(name) for name in names]
    if any(sub is None for sub in subs):
        # return self
        raise LookupError("One of the names is missing from the diagram")
    else:
        # NOTE Unfortunately, mypy is not narrowing the type when using the
        # `any` or `all` functions.
        # https://github.com/python/mypy/issues/13069
        # Hopefully this bug will be fixed at some point in the future.
        return f(subs, self)  # type: ignore


@dataclass
class SubMap(Monoid):
    data: Dict[Name, List[Subdiagram]]

    def __add__(self, other: SubMap) -> SubMap:
        d1 = self.data
        d2 = other.data
        return SubMap(
            {k: d1.get(k, []) + d2.get(k, []) for k in set(d1) | set(d2)}
        )

    @classmethod
    def empty(cls) -> SubMap:
        return SubMap({})


class GetSubMap(DiagramVisitor[SubMap, Affine]):
    A_type = SubMap

    def visit_apply_transform(
        self,
        diagram: ApplyTransform,
        t: Affine
    ) -> SubMap:
        return diagram.diagram.accept(self, t * diagram.transform)

    def visit_apply_name(
        self,
        diagram: ApplyName,
        t: Affine
    ) -> SubMap:
        d1 = SubMap({diagram.dname: [Subdiagram(diagram.diagram, t)]})
        d2 = diagram.diagram.accept(self, t)
        return d1 + d2


def get_sub_map(
    self: Diagram, t: Affine
) -> Dict[Name, List[Subdiagram]]:
    """Retrieves all named subdiagrams in the given diagram and accumulates
    them in a dictionary (map) indexed by their name.
    """
    return self.accept(GetSubMap(), t).data
