# + tags=["hide"]
from chalk.core import BaseDiagram
from chalk import *

def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
# -


# Elementary diagrams can be created using shapes and paths.

# ## Polygons

# ### triangle

# + tags=["hide_inp"]
help(triangle)
# -

#

triangle(1)


# ### square

# + tags=["hide_inp"]
help(square)
# -

#

square(1)

# ### rectangle

# + tags=["hide_inp"]
help(rectangle)
# -

rectangle(3, 1, 0.0)


# ### polygon

# + tags=["hide_inp"]
help(regular_polygon)
# -

regular_polygon(8, 2)

# ### Circle-like shapes

# + tags=["hide_inp"]
help(circle)
# -


circle(1)

# ### arc

# + tags=["hide_inp"]
help(arc)
# -

#

quarter = 90
arc(1, 0, quarter) 

#

arc(1, 0, quarter) + arc(1, 2 * quarter, 3 * quarter) 


# ### arc_between

# + tags=["hide_inp"]
help(arc_between)
# -

#

arc_between((0, 0), (1, 0), 1)


# ### Text

# + tags=["hide_inp"]
help(text)
# -

# Note that unlike other shapes, ``text`` has an empty envelope, so we need to explicitly specify it in order to get a non-empty rendering.

text("hello", 1).with_envelope(rectangle(2.5, 1))

# ### Paths

# + tags=["hide_inp"]
help(make_path)
# -

#

make_path([(0, 0), (0, 1), (1, 1), (1, 2)])


