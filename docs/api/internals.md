---
title: Chalk internals
summary: A look inside Chalk
date: 2022-08-21
---

This document presents information on the internal implementation of the Chalk library.
This information should not be needed by the casual user of the library, but it can certainly be useful to developers and also provides a record of the major design decisions.
While much of the functionality of the Chalk library resembles the [Haskell `diagrams` library](https://diagrams.github.io/),
there are important distinctions in the implementation due to the differences of the two languages (Python and Haskell).

## Core data types

Chalk is an embedded domain specific language (EDSL).
The two extremes of language design are shallow and deep EDSLs.
Loosely put, a shallow EDSL specifies the language through a set of functions,
while a deep EDSL specifies the syntax of the language using a data type (the abstract syntax tree; AST), which is then interpreted using given evaluator functions.
Chalk uses a hybrid approach which defines an intermediate core data structure (in our case, the `Diagram` type) and a suite of functions that operate on this type.
(For more information on the concepts of deep and shallow EDSLs, see [the paper of Gibbons and Wu from ICFP'14](http://www.cs.ox.ac.uk/jeremy.gibbons/publications/embedding.pdf)).

The `Diagram` type (implemented as `BaseDiagram` in `chalk/core.py`) can be thought as an [algebraic data type](https://en.wikipedia.org/wiki/Algebraic_data_type) with the following six variants:
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

Instances of `BaseDiagram` can be constructed and modified using the functions provided by the library.
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
other interpretations are flattening the `Diagram` AST to a list of `Primitive`s or extracting the `Envelope` or `Trace` of a `Diagram`.
All these functions are implemented using the [visitor pattern](https://en.wikipedia.org/wiki/Visitor_pattern#Python_example), which is the object-oriented correspondent of pattern matching and folding encountered in functional programming.
(Jeremy Gibbons provides a nice exposition on the relationship between object-oriented design patterns and their functional counterparts in his paper [Design Patterns as Higher-Order Datatype-Generic Programs](http://www.cs.ox.ac.uk/jeremy.gibbons/publications/hodgp.pdf)).

## Support for functional and object-oriented use

Internally the library is implemented in a functional style, through functions that operate on the `Diagram` AST.
For example, for composition we have a combinator function `beside` (in `chalk/combinators.py`) that takes two `Diagram`s and returns a new `Diagram` corresponding to their composition.
The main benefit of using functions is that the library can be easily split into self-contained submodules, each pertaining to a certain type of functionality (shapes, alignment, transformations, and so on).
In contrast, using an object-oriented style would bundle the entire functionality as methods inside the class
and would only allow to separate the variants (e.g., `Empty`, `Primitive`, `Compose`) across files (see the ["expression problem"](https://en.wikipedia.org/wiki/Expression_problem) for the trade-offs between the functional and object-oriented style).

However, allowing to write code in an object-oriented style (using dot notation) provides an arguably more convenient and idiomatic style in Python.
For this reason, we attach all the functions as methods in the `chalk/core.py` file; for example:
```python
class BaseDiagram:
    beside = chalk.combinators.beside
```
This implementation allows writing
```python
circle(1).beside(circle(1), unit_x)
```
and even more succinctly
```python
circle(1) | circle(1)
```
as we define ["dunder" methods](https://docs.python.org/3/reference/datamodel.html#special-method-names) for common combinators (`__or__` in this case).

The challenge of this implementation decision is that it complicates the [type checking](https://mypy.readthedocs.io/en/stable/index.html#) due to circular imports.
We solve this problem by introducing a `Diagram` [protocol](https://mypy.readthedocs.io/en/stable/protocols.html) (in `chalk/types.py`), which specifies type signatures for all the methods that are to be implemented later on.
The `Diagram` type is used throughout the code for type hinting, for example:
```python
def juxtapose(self: Diagram, other: Diagram, direction: V2) -> Diagram:
    # We can use `get_envelope` here since the `Diagram` protocol promises
    # that such a method will be implemented.
```
The concrete implementation of the `Diagram` protocol is provided by the `BaseDiagram` in `chalk/core.py`.

## Trail-like data types

Apart from the main `Diagram` data type, there are several other related types (`Trail`, `Located`, `Path`) that encode a trail-like drawing and can be "lifted" to a `Diagram` using the `stroke` method.
These trail-like structures encode different types of information and, as a consequence, have different combination semantics:

- `Trail` corresponds to a list of vectors (translation-invariant offsets, which can be either straight or bendy—implemented as arcs).
A `Trail` is `Transformable` (by transforming each of the vectors), but since vectors are translation-invariant, applying `translate` leaves a `Trail` unchanged.
The monoid composition corresponds to the list monoid: it _extends_ the first trail with the second one.
A `Trail` can be closed which means that it is a loop and it will be able to hold a color when filled in.
- `Located` is a `Trail` paired with a `location` origin point.
A `Trail` can be turned into `Located` using the `at` method which specifies the origin location.
`Located` instances do not form a monoid.
- `Path` is a list of `Located` instances.
Having more than one `Located` instance is important, since it allows to easily draw objects with holes in them (such as rings).
The monoid composition also corresponds to the list monoid, but the effects is different from `Trail`: it _overlays_ the `Located` subpaths.

Below we present an example (inspired from the `diagrams` library) that showcases the distinction of the combination semantics (the `concat` function) for the `Diagram`, `Path`, `Trail` types.

```python
from colour import Color
from toolz import iterate, take
from chalk import *

red = Color("red")
t = Trail.regular_polygon(3, 1)
t_loc = t.centered()

# Diagram
dia1 = concat(take(3, iterate(lambda d: d.rotate_by(1 / 9), t_loc.stroke()))).fill_color(red)

# Path
dia2 = Path.concat(take(3, iterate(lambda d: d.rotate_by(1 / 9), t_loc.to_path()))).stroke().fill_color(red)

# Trail
dia3 = Trail.concat(take(3, iterate(lambda d: d.rotate_by(1 / 9), t))).stroke().center_xy()

hcat([dia1, dia2, dia3], sep=0.2)
```

<img src="https://user-images.githubusercontent.com/819256/184515950-b3ce4245-19ee-4357-bc3a-f32f993b04ef.png">
