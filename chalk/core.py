from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, List, Optional, TypeVar

import chalk.align
import chalk.arrow
import chalk.backend.cairo
import chalk.backend.svg
import chalk.backend.tikz
import chalk.combinators
import chalk.model
import chalk.subdiagram
import chalk.trace
import chalk.types
from chalk import backend
from chalk.envelope import Envelope
from chalk.style import Style, StyleHolder
from chalk.subdiagram import Name
from chalk.transform import Affine, unit_x
import chalk.transform as tx
from chalk.types import Diagram, Shape
from chalk.utils import imgen
from chalk.visitor import DiagramVisitor
from chalk.monoid import MList

Trail = Any
Ident = tx.ident
A = TypeVar("A", bound=chalk.monoid.Monoid)

SVG_HEIGHT = 200
SVG_DRAW_HEIGHT = None


def set_svg_height(height: int) -> None:
    "Globally set the svg preview height for notebooks."
    global SVG_HEIGHT
    SVG_HEIGHT = height


def set_svg_draw_height(height: int) -> None:
    "Globally set the svg preview height for notebooks."
    global SVG_DRAW_HEIGHT
    SVG_DRAW_HEIGHT = height


@dataclass
class BaseDiagram(chalk.types.Diagram):
    """Diagram class."""

    # Monoid
    __add__ = chalk.combinators.atop

    def __rmatmul__(self, t: Affine) -> Diagram:
        return self.apply_transform(t)

    @classmethod
    def empty(cls) -> Diagram:  # type: ignore
        return Empty()

    # Tranformable
    def apply_transform(self, t: Affine) -> Diagram:  # type: ignore
        if t.shape[0] == 1:
           return ApplyTransform(t, self)
        else:
            return chalk.combinators.concat(
                [ApplyTransform(t[i][None], self)
                for i in range(t.shape[0])])
    # Stylable
    def apply_style(self, style: Style) -> Diagram:  # type: ignore
        return ApplyStyle(style, self)

    def _style(self, style: Style) -> Diagram:
        return self.apply_style(style)

    def compose(
        self, envelope: Envelope, other: Optional[Diagram] = None
    ) -> Diagram:
        if other is None and isinstance(self, Compose):
            return Compose(envelope, self.diagrams)
        if other is None and isinstance(self, Compose):
            return Compose(envelope, [self])
        
        other = other if other is not None else Empty() 
        if isinstance(self, Empty):
            return other
        elif isinstance(self, Compose) and isinstance(other, Compose):
            return Compose(envelope, self.diagrams + other.diagrams)
        elif isinstance(other, Empty) and not isinstance(self, Compose):
            return Compose(envelope, [self])

        elif isinstance(other, Compose):
            return Compose(envelope, [self] + other.diagrams)
        else:
            return Compose(envelope, [self, other])

    def named(self, name: Name) -> Diagram:
        """Add a name (or a sequence of names) to a diagram."""
        return ApplyName(name, self)

    # Combinators
    with_envelope = chalk.combinators.with_envelope
    close_envelope = chalk.combinators.close_envelope
    juxtapose = chalk.combinators.juxtapose
    juxtapose_snug = chalk.combinators.juxtapose_snug
    beside_snug = chalk.combinators.beside_snug
    above = chalk.combinators.above
    atop = chalk.combinators.atop
    beside = chalk.combinators.beside
    above = chalk.combinators.above

    # Align
    align = chalk.align.align_to
    align_t = chalk.align.align_t
    align_b = chalk.align.align_b
    align_l = chalk.align.align_l
    align_r = chalk.align.align_r
    align_tr = chalk.align.align_tr
    align_tl = chalk.align.align_tl
    align_bl = chalk.align.align_bl
    align_br = chalk.align.align_br
    center_xy = chalk.align.center_xy
    center = chalk.align.center
    scale_uniform_to_y = chalk.align.scale_uniform_to_y
    scale_uniform_to_x = chalk.align.scale_uniform_to_x

    # Arrows
    connect = chalk.arrow.connect
    connect_outside = chalk.arrow.connect_outside
    connect_perim = chalk.arrow.connect_perim

    # Model
    show_origin = chalk.model.show_origin
    show_envelope = chalk.model.show_envelope
    show_beside = chalk.model.show_beside
    show_labels = chalk.model.show_labels

    # Combinators
    frame = chalk.combinators.frame
    pad = chalk.combinators.pad

    # Infix
    def __or__(self, d: Diagram) -> Diagram:
        return chalk.combinators.beside(self, d, unit_x)

    __truediv__ = chalk.combinators.above
    __floordiv__ = chalk.combinators.above2

    def display(
        self, height: int = 256, verbose: bool = True, **kwargs: Any
    ) -> None:
        """Display the diagram using the default renderer.

        Note: see ``chalk.utils.imgen`` for details on the keyword arguments.
        """
        # update kwargs with defaults and user-specified values
        kwargs.update({"height": height})
        kwargs.update({"verbose": verbose})
        kwargs.update({"dirpath": None})
        kwargs.update({"wait": kwargs.get("wait", 1)})
        # render and display the diagram
        imgen(self, **kwargs)

    # Rendering
    render = chalk.backend.cairo.render
    render_png = chalk.backend.cairo.render
    render_svg = chalk.backend.svg.render
    render_pdf = chalk.backend.tikz.render

    to_svg = backend.svg.to_svg
    to_tikz = backend.tikz.to_tikz

    def _repr_svg_(self) -> str:
        global SVG_HEIGHT
        f = tempfile.NamedTemporaryFile(delete=False)
        self.render_svg(f.name, height=SVG_HEIGHT, draw_height=SVG_DRAW_HEIGHT)
        f.close()
        svg = open(f.name).read()
        os.unlink(f.name)
        return svg

    def _repr_html_(self) -> str | tuple[str, Any]:
        """Returns a rich HTML representation of an object."""
        return self._repr_svg_()

    # Getters
    get_envelope = chalk.envelope.get_envelope
    get_trace = chalk.trace.get_trace
    get_subdiagram = chalk.subdiagram.get_subdiagram
    get_sub_map = chalk.subdiagram.get_sub_map

    with_names = chalk.subdiagram.with_names

    def qualify(self, name: Name) -> Diagram:
        """Prefix names in the diagram by a given name or sequence of names."""
        return self.accept(Qualify(name), None)

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        raise NotImplementedError


@dataclass
class Primitive(BaseDiagram):
    """Primitive class.

    This is derived from a ``chalk.core.Diagram`` class.

    [TODO]: explain what Primitive class is for.
    """

    shape: Shape
    style: StyleHolder
    transform: Affine

    @classmethod
    def from_shape(cls, shape: Shape) -> Primitive:
        """Creates a primitive from a shape using the default style (only line
        stroke, no fill) and the identity transformation.

        Args:
            shape (Shape): A shape object.

        Returns:
            Primitive: A diagram object.
        """
        return cls(shape, StyleHolder.empty(), Ident)

    def apply_transform(self, t: Affine) -> Primitive:
        if t.shape[0] != 1:
            return super().apply_transform(t)

        if hasattr(self.transform, "shape"):
            new_transform = t @ self.transform
        else:
            new_transform = t
        return Primitive(self.shape, self.style, new_transform)

    def apply_style(self, other_style: Style) -> Primitive:
        """Applies a style and returns a primitive.

        Args:
            other_style (Style): A style object.

        Returns:
            Primitive
        """
        return Primitive(
            self.shape, self.style.merge(other_style), self.transform
        )

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_primitive(self, args)


@dataclass
class Empty(BaseDiagram):
    """An Empty diagram class."""

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_empty(self, args)

    def apply_transform(self, t: Affine) -> Empty:
        return Empty()

    def apply_style(self, t: Affine) -> Empty:
        return Empty()

@dataclass
class Compose(BaseDiagram):
    """Compose class."""

    envelope: Envelope
    diagrams: List[Diagram]

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_compose(self, args)

@dataclass
class MultiDiagram(BaseDiagram):
    diagrams: List[Diagram]

    def __iter__(self):
        return self.diagrams.__iter__()

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_apply_multi(self, args)


@dataclass
class ApplyTransform(BaseDiagram):
    """ApplyTransform class."""

    transform: Affine
    diagram: Diagram

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_apply_transform(self, args)

    def apply_transform(self, t: Affine) -> ApplyTransform:
        if t.shape[0] != 1:
            return super().apply_transform(t)

        new_transform = t @ self.transform
        return ApplyTransform(new_transform, self.diagram)
    
@dataclass
class ApplyStyle(BaseDiagram):
    """ApplyStyle class."""

    style: Style
    diagram: Diagram

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_apply_style(self, args)

    def apply_style(self, style: Style) -> ApplyStyle:
        new_style = style.merge(self.style)
        return ApplyStyle(new_style, self.diagram)

@dataclass
class ApplyName(BaseDiagram):
    """ApplyName class."""

    dname: Name
    diagram: Diagram

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_apply_name(self, args)


class Qualify(DiagramVisitor[Diagram, None]):
    A_type = Diagram

    def __init__(self, name: Name):
        self.name = name

    def visit_primitive(self, diagram: Primitive, args: None) -> Diagram:
        return diagram

    def visit_compose(self, diagram: Compose, args: None) -> Diagram:
        return Compose(
            diagram.envelope, [d.accept(self, None) for d in diagram.diagrams]
        )

    def visit_apply_transform(
        self, diagram: ApplyTransform, args: None
    ) -> Diagram:
        return ApplyTransform(
            diagram.transform,
            diagram.diagram.accept(self, None),
        )

    def visit_apply_style(self, diagram: ApplyStyle, args: None) -> Diagram:
        return ApplyStyle(
            diagram.style,
            diagram.diagram.accept(self, None),
        )

    def visit_apply_name(self, diagram: ApplyName, args: None) -> Diagram:
        return ApplyName(
            self.name + diagram.dname, diagram.diagram.accept(self, None)
        )


@dataclass
class FlatPrimitive:
    "A version of primitive that may be batched"
    shape: Path
    style: StyleHolder
    transform: Affine

    @staticmethod
    def from_primitive(self):
        return FlatPrimitive(self.shape, self.style, self.transform)


def flatten(diagram: Diagram):
    children = [FlatPrimitive(d.shape, d.style, d.transform) 
                for d in diagram.accept(ToList(), Ident)]
    return children
    
def unf(diagram):
    from chalk.combinators import concat
    return concat([Primitive(d.shape, 
                             d.style,
                             d.transform) 
                    for d in diagram]), None
def unflatten(diagrams):
    from chalk.combinators import concat
    return [concat([Primitive(d.shape.split(i), 
                             d.style.split(i), 
                             d.transform[i]) 
                        for d in diagrams])
            for i in range(diagrams[0].transform.shape[0])]

# def unflatten(aux_data, diagrams):
#     from chalk.combinators import concat
#     return concat([Primitive(d.shape.split(i), 
#                               d.style.split(i), 
#                               d.transform[i]) 
#                     for d in diagrams])
#             for i in range(diagrams[0].transform.shape[0])]

#import jax
# from jax.tree_util import register_pytree_node

# for x in [MultiDiagram, Primitive, Compose, ApplyName, ApplyStyle, ApplyTransform]:
#     register_pytree_node(
#         x,
#         flatten,    # tell JAX what are the children nodes
#         unflatten   # tell JAX how to pack back into a RegisteredSpecial
#     )

class ToList(DiagramVisitor[MList[FlatPrimitive], Affine]):
    """Compiles a `Diagram` to a list of `FlatPrimitive`s. The transformation `t`
    is accumulated upwards, from the tree's leaves.
    """

    A_type = MList[Primitive]

    def visit_primitive(
        self, diagram: Primitive, t: Affine = Ident
    ) -> MList[Primitive]:
        return MList([(diagram.apply_transform(t))])

    def visit_apply_multi(
        self, diagram, t: Affine = Ident
    ) -> MList[Primitive]:
        return MList(diagram.diagrams)


    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine = Ident
    ) -> MList[Primitive]:
        if hasattr(diagram.transform, "shape"):
            t_new = t @ diagram.transform
        else: 
            t_new = t
        return MList(
            [
                prim.apply_transform(t_new)
                for prim in diagram.diagram.accept(self, t).data
            ]
        )

    def visit_apply_style(
        self, diagram: ApplyStyle, t: Affine = Ident
    ) -> MList[Primitive]:
        return MList(
            [
                prim.apply_style(diagram.style)
                for prim in diagram.diagram.accept(self, t).data
            ]
        )

    def visit_apply_name(
        self, diagram: ApplyName, t: Affine = Ident
    ) -> MList[Primitive]:
        return MList([prim for prim in diagram.diagram.accept(self, t).data])
