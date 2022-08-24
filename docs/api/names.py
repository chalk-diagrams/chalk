# + tags=["hide_inp"]
from chalk.core import BaseDiagram
from chalk import *

def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
# -

# Chalk supports basic methods for complex
# connected layouts and diagrams. Individual elements
# can be assigned names, and then be referenced in
# their subdiagram locations. 

# ### Diagram.named

# + tags=["hide_inp"]
help(BaseDiagram.named)
# -

diagram = circle(0.5).named("x") | square(1)
diagram

# ### place_on_path

# + tags=["hide_inp"]
help(place_on_path)
# -

place_on_path(
    [circle(0.25) for _ in range(6)],
    Trail.regular_polygon(6, 1).to_path(),
)

# ### Diagram.get_subdiagram_envelope

# + tags=["hide_inp"]
help(BaseDiagram.get_subdiagram_envelope)
# -

#

diagram = circle(0.5).named("x") | square(1) 
bbox = diagram.get_subdiagram_envelope("x")
diagram + circle(0.2).translate(bbox.center.x, bbox.center.y)

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
