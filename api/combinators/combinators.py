# + tags=["hide_inp"]
import math
from planar import Vec2
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

# ### beside

# + tags=["hide_inp"]
help(beside)
# -

#

diagram = triangle(1).beside(square(1), unit_x)
diagram

#

triangle(1).show_beside(square(1), unit_x)

triangle(1).show_beside(triangle(1).rotate_by(1 / 6), Vec2.polar(-45))

#

triangle(1).show_beside(triangle(1).rotate_by(1 / 6), Vec2.polar(-30))


# ### above

# + tags=["hide_inp"]
help(above)
# -

#

diagram = triangle(1) / square(1)
diagram

#

diagram.show_envelope().show_origin()


# ### atop

# + tags=["hide_inp"]
help(atop)
# -

# Example 1 - Atop at origin

diagram = square(1) + triangle(1)
diagram

#

diagram.show_envelope().show_origin()

# Example 2 - Align then atop origin

s = square(1).align_r().align_b().show_origin()
t = triangle(1).align_l().align_t().show_origin()
s

#

t

#

s + t



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
