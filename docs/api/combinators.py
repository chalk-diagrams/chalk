# + tags=["hide_inp"]
import math
from chalk import *
def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
# -


# Complex diagrams can be created by combining simpler diagrams
# through placement combinators. These place diagrams above, atop or
# besides other diagrams. Relative location is determined by the bounding box
# and origins of the diagrams.

# ### above

# + tags=["hide_inp"]
help(above)
# -

#

diagram = triangle(1) / square(1)
diagram

#

diagram.show_bounding_box().show_origin()


# ### atop

# + tags=["hide_inp"]
help(atop)
# -

# Example 1 - Atop at origin

diagram = square(1) + triangle(1)
diagram

#

diagram.show_bounding_box().show_origin()

# Example 2 - Align then atop origin

s = square(1).align_r().align_b().show_origin()
t = triangle(1).align_l().align_t().show_origin()
s

#

t

#

s + t

# ### beside

# + tags=["hide_inp"]
help(beside)
# -

#

diagram = triangle(1) | square(1)
diagram

#

triangle(1).show_origin() | square(1).show_origin()

#

diagram.show_origin()


# ### vcat

# + tags=["hide_inp"]
help(vcat)
# -

#

vcat([triangle(1), square(1), triangle(1)], 0.2)

# ### concat

# + tags=["hide_inp"]
help(concat)
# -

#

concat([triangle(1), square(1), triangle(1)])


# ### hcat

# + tags=["hide_inp"]
help(hcat)
# -

#

hcat([triangle(1), square(1), triangle(1)], 0.2)
