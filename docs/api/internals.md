---
title: Chalk internals
summary: A look inside Chalk
date: 2022-08-21
---

This document presents information pertaining to the internal implementation of the Chalk library.
This information should not be needed for the user of the library, but it might be useful for developers and also provides a recording of the major design decisions.

**Core data types.**
Chalk is an embedded domain specific language (EDSL).
The two extremes of language design are shallow and deep EDSLs.
Loosely put, a shallow EDSL specifies the language through a set of functions,
while a deep EDSL provides a data type (which defines the syntax of the language) and evaluator functions that interpret a given abstract syntax tree (AST).
Chalk uses a hybrid approach which defines an intermediate (small) data structure (in our case, the `Diagram` type) and a suite of functions that operate on this type.
(For more information on the concepts of deep, shallow and intermediate representations for EDSL, see [the paper of Gibbons and Wu from ICFP'14](http://www.cs.ox.ac.uk/jeremy.gibbons/publications/embedding.pdf)).

The `Diagram` type (implemented as `BaseDiagram` in `chalk/core.py`) can be thought as an [algebraic data type](https://en.wikipedia.org/wiki/Algebraic_data_type) with the following variants:
`Empty`, `Primitive`, `Compose`, `ApplyTransform`, `ApplyStyle`, `ApplyName`.
Each of the variants may hold additional information, as follows:

```python
class Empty(BaseDiagram):
    pass

class Primitive(BaseDiagram):
    shape: Shape
    style: Style
    transform: Affine

class Compose(BaseDiagram):
    envelope: Envelope
    diagram1: Diagram
    diagram2: Diagram

class ApplyTransform(BaseDiagram):
    transform: Affine
    diagram: Diagram

class ApplyStyle(BaseDiagram):
    style: Style
    diagram: Diagram

class ApplyName(BaseDiagram):
    dname: str
    diagram: Diagram
```

Instances of `BaseDiagram` can be constructed and modified using the provided functions.
For example,
```python
circle(1)
```
generates the following AST:
```python
Primitive(shape=Path(...), style=Style(...), transform=Affine(...))
```
and
```python
circle(1) | circle(1)
```
generates the following AST:
```python
Compose(
    envelope=...,
    diagram1=Primitive(shape=Path(...), style=Style(...), transform=Affine(...)),
    diagram2=Primitive(shape=Path(...), style=Style(...), transform=Affine(...)),
)
```
A `Diagram` AST can be interpreted in multiple ways, arguably the most obvious being through the rendering functions (see the `chalk/backend` submodule);
other interpretations are compiling the `Diagram` to a list of primitives or extracting the `Envelope` or `Trace` of a `Diagram`.
These interpreting functions are implemented using the [visitor pattern](https://en.wikipedia.org/wiki/Visitor_pattern#Python_example), which is the object-oriented equivalent of pattern matching and folding encountered in functional programming.

**Functional and object-oriented style.**
Internally the library is implemented in a functional style, through functions that operate on the `Diagram` AST.
For example, for composition we have a combinator function `beside` (in `chalk/combinators.py`) that takes two `Diagrams` and returns their composition: one diagram placed next to the other.
The main benefit of having functions is that the library can be eas split into self-contained submoduled;
instead of having all the functionality in methods, which would results in a large file.
However, for a more convenient syntax in Python, we attach all these functions as _methods_ in the `chalk/core.py` file;
for example:
```python
class BaseDiagram:
    beside = chalk.combinators.beside
```
This implementation allows writing:
```python
circle(1).beside(circle(1), unit_x)
```
and even more succinctly
```python
circle(1) | circle(1)
```
as we define ["dunder" methods](https://docs.python.org/3/reference/datamodel.html#special-method-names) for common combinators (`__or__` in this case).

In order to type check this sort of style, we had introduce an interface `Diagram` (in `chalk/types.py`), which specifies all the type signatures of the methods to be implemented.
The `Diagram` type is used throughout the code, for example:
```python
def beside(self: Diagram, other: Diagram, direction: V2) -> Diagram:
    ...
```
The concrete implementation is deferred to the `BaseDiagram` in `chalk/core.py`.

**Related data types.**
Apart from the main `Diagram` data type, there are several other related types (`Trail`, `Located`, `Path`) the share similarities to the main type and can be "lifted" to it using the `stroke` method.
An important distinction is the combination semantics (that is, the [monoid](https://en.wikipedia.org/wiki/Monoid) instances).
- `Trail` is a list of translation-invariant offsets.
These are `Transformable`, but given that they are translation-invariant applying `translate` leaves a `Trail` unchanged.
The monoid composition corresponds to the list monoid: extending the first trail with the second one.
A `Trail` can be closed which means that it is a loop, so it will have a color when filled in.
- `Located` is a `Trail` with a `location` origin point. A `Trail` can be turned into `Located` using the `at` method which specifies the origin location.
- `Path` is a list of `Located` instances. The monoid composition also corresponds to the list monoid.

An examples that showcases the distinction of the various monoid instances is the following.

```python
from colour import Color
from toolz import iterate, take
from chalk import *

red = Color("red")
t = Trail.regular_polygon(3, 1)
t_loc = t.centered()

dia1 = concat(take(3, iterate(lambda d: d.rotate_by(1 / 9), t_loc.stroke()))).fill_color(red)
dia2 = Path.concat(take(3, iterate(lambda d: d.rotate_by(1 / 9), t_loc.to_path()))).stroke().fill_color(red)
dia3 = Trail.concat(take(3, iterate(lambda d: d.rotate_by(1 / 9), t))).stroke().center_xy()

dia_all = hcat([dia1, dia2, dia3], sep=0.2)
dia_all
```
