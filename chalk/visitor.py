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
    from chalk.monoid import Monoid
    from chalk.Path import Path
    from chalk.shapes import Image, Latex, Spacer, Text

    A = TypeVar("A", bound=Monoid)
else:
    A = TypeVar("A")

B = TypeVar("B")

from itertools import product

def to_size(index, shape):
    if len(index) > len(shape):
        index = index[len(index)-len(shape):]
    return tuple(i if s > 1 else 0 for s, i in zip(shape, index))

def remove(index, axis):
    return tuple(i for j, i in enumerate(index) if j != axis)

def rproduct(rtops):
    return product(*map(range, rtops))

class DiagramVisitor(Generic[A, B]):
    A_type: type[A]

    def collapse_array(self):
        return True

    def visit_primitive_array(self, diagram: Primitive, arg: B) -> A:
        size = diagram.size()
        return {key: self.visit_primitive(diagram[key], arg[key])
                for key in rproduct(size)}

    def visit_primitive(self, diagram: Primitive, arg: B) -> A:
        "Primitive defaults to empty"
        return self.A_type.empty()

    def visit_empty(self, diagram: Empty, arg: B) -> A:
        "Empty defaults to empty"
        return self.A_type.empty()

    def visit_compose_array(self, diagram: Compose, arg: B) -> A:
        print("HERE!")
        size = diagram.size()
        ret = {key: self.A_type.empty() 
               for key in rproduct(size)}
        for d in diagram.diagrams:
            print("d")
            d_size = d.size()
            a = d.accept(self, arg)
            ret = {k: ret[k] + a[to_size(k, d_size)] 
                    for k in rproduct(size)}
        return ret

    def visit_compose(self, diagram: Compose, arg: B) -> A:
        "Compose defaults to monoid over children"
        return self.A_type.concat(
                [d.accept(self, arg) for d in diagram.diagrams]
            )

    def visit_compose_axis(self, diagram: ComposeAxis, t:V2_t) -> EnvDistance:
        if not self.collapse_array():
            import jax
            from functools import partial
            axis = len(diagram.diagrams.size()) -1
            print(axis)
            fn = diagram.diagrams.accept.__func__
            ed = jax.vmap(partial(fn, visitor=self, args=t),
                          in_axes=axis, out_axes=axis)(diagram.diagrams)
            return ed.reduce(axis)
        else:
            "Compose defaults to monoid over children"
            size = diagram.size()
            internal_size = diagram.diagrams.size()
            print("calling", diagram.diagrams)
            internal = diagram.diagrams.accept(self, t[..., None, :, :, :])
            print(internal_size)
            ret = {key:[] for  key in rproduct(size)}
            print(internal.keys())
            if size == ():
                ret[()] = []

            for key in rproduct(internal_size):
                ret[key[:-1]].append(internal[key])
            for key in ret:
                ret[key] = self.A_type.concat(ret[key])
            if size == ():
                return ret[()]
            return ret

    def visit_apply_transform_array(self, diagram: Primitive, arg: B) -> A:
        return self.visit_apply_transform(diagram, arg)

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
