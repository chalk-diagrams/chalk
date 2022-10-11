# + tags=["hide_inp"]
from colour import Color
from chalk.core import BaseDiagram
from chalk import *

def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))

set_svg_height(100)
# -

# Chalk supports basic methods for complex connected layouts and diagrams.
# Individual elements can be assigned names, and then be referenced in their subdiagram locations.
# As we will see, names are particularly useful for connecting different parts of the diagram with arrows.
# Our implementation follows the API of the Haskell [diagrams](https://diagrams.github.io/doc/manual.html#named-subdiagrams) library,
# but named nodes are also common in TikZ.
#
# ### Diagram.named

# + tags=["hide_inp"]
help(BaseDiagram.named)
# -

diagram = circle(0.5).named("x") | square(1)
diagram

# ### Diagram.get_subdiagram

# + tags=["hide_inp"]
help(BaseDiagram.get_subdiagram)
# -

# A `Subdiagram` is a `Diagram` paired with its enclosing context (a `Transformation` for the moment; but `Style` should also be added at some point).
# It has the following methods:
# - `get_envelope`, which returns the corresponding `Envelope`
# - `get_trace`, which returns the corresponding `Trace`
# - `get_location`, which returns the local origin of the `Subdiagram`
# - `boundary_from`, which return the furthest point on the boundary of the `Subdiagram`, starting from the local origin of the `Subdigram` and going in the direction of a given vector.

#

diagram = circle(0.5).named("x") | square(1)
sub = diagram.get_subdiagram("x")
diagram + circle(0.2).translate_by(sub.get_location())

# ### Diagram.with_names

# + tags=["hide_inp"]
help(BaseDiagram.with_names)
# -

root = circle(1).named("root")
leaves = hcat([circle(1).named(c) for c in "abcde"], sep=0.5).center()

def connect(subs, nodes):
    root, leaf = subs
    pp = tuple(root.boundary_from(unit_y))
    pc = tuple(leaf.boundary_from(-unit_y))
    return nodes + make_path([pp, pc])

nodes = root / vstrut(2) / leaves

for c in "abcde":
    nodes = nodes.with_names(["root", c], connect)
nodes

# ### Diagram.qualify

# + tags=["hide_inp"]
help(BaseDiagram.qualify)
# -

red = Color("red")

def attach(subs, dia):
    sub1, sub2 = subs
    p1 = tuple(sub1.get_location())
    p2 = tuple(sub2.get_location())
    return dia + make_path([p1, p2]).line_color(red)


def squares():
    s = square(1)
    return (
        (s.named("NW") | s.named("NE")) /
        (s.named("SW") | s.named("SE")))


dia = hcat([squares().qualify(str(i)) for i in range(5)], sep=0.5)
pairs = [
    (("0", "NE"), ("2", "SW")),
    (("1", "SE"), ("4", "NE")),
    (("3", "NW"), ("3", "SE")),
    (("0", "SE"), ("1", "NW")),
]

dia

for pair in pairs:
    dia = dia.with_names(pair, attach)

dia

# ### Diagram.show_labels

# + tags=["hide_inp"]
help(BaseDiagram.show_labels)
# -

dia.show_labels(font_size=0.2)

# ### Diagram.connect

# + tags=["hide_inp"]
help(BaseDiagram.connect)
# -

diagram = circle(0.5).named("x") | hstrut(1) | square(1).named("y")
diagram.connect("x", "y")

# ### Diagram.connect_outside

# + tags=["hide_inp"]
help(BaseDiagram.connect_outside)
# -

diagram = circle(0.5).named("x") | hstrut(1) | square(1).named("y")
diagram.connect_outside("x", "y")
