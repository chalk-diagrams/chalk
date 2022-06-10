# + tags=["hide"]
import math
from chalk import *
def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
# -

# ### circle

# + tags=["hide_inp"]
help(circle)
# -


circle(1) + circle(0.5)

# ### arc

# + tags=["hide_inp"]
help(arc)
# -

#

quarter = math.pi / 2
arc(1, 0, quarter) 

#

arc(1, 0, quarter) + arc(1, 2 * quarter, 3 * quarter) 


# ### arc_between

# + tags=["hide_inp"]
help(arc_between)
# -

#

arc_between((0, 0), (1, 0), 1)

# ### polygon

# + tags=["hide_inp"]
help(polygon)
# -

#

polygon(8, 2)


# ### square

# + tags=["hide_inp"]
help(square)
# -

#

square(1)

# ### triangle

# + tags=["hide_inp"]
help(triangle)
# -

#

triangle(1)


# ### rectangle

# + tags=["hide_inp"]
help(rectangle)
# -

#

rectangle(8, 2, 0.5)



# ### make_path

# + tags=["hide_inp"]
help(make_path)
# -

#

make_path([(0, 0), (0, 1), (1, 1), (1, 2)])

# ### text

# + tags=["hide_inp"]
help(text)
# -

#

text("hello", 0.2)

#

text("hello", 0.2).show_bounding_box()
