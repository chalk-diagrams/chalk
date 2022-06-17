# + tags=["hide_inp"]
import math
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
help(Diagram.named)
# -

#

diagram = triangle(1).named("x") | square(1)
diagram

# ### place_on_path

# + tags=["hide_inp"]
help(place_on_path)
# -

#

place_on_path([triangle(0.4) for i in range(5)],
              Path.from_list_of_tuples([(i, i % 2) for i in range(5)]))

# ### Diagram.get_subdiagram_bounding_box

# + tags=["hide_inp"]
help(Diagram.get_subdiagram_bounding_box)
# -

#

diagram = triangle(1).named("x") | square(1) 
bbox = diagram.get_subdiagram_bounding_box("x")
diagram + place_on_path([triangle(0.4)], Path.from_point(bbox.center))

# ### connect

# + tags=["hide_inp"]
help(connect)
# -

#

diagram = triangle(1).named("x") / square(1).named("y")
diagram + connect(diagram, "x", "y")

# ### connect_outer

# + tags=["hide_inp"]
help(connect_outer)
# - 

#

diagram = square(1).named("x") | hstrut(1) | square(1).named("y")
diagram + connect_outer(diagram, "x", "E", "y", "W")
