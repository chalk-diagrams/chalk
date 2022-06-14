# + tags=["hide_inp"]
import math
from chalk import *
def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
# -

# Chalk also includes a Trail primitive which allows for creating
# new Diagrams and more complex shapes. Trails are specified
# by position-invariant offsets and can be rendered into paths.
# See the [Koch](../examples/koch/) example for a use case. 


# ### Trail

# Trails are a sequence of vectors.

trail = Trail([Vector(1, 0), Vector(1, 1), Vector(0, 1)])

# ### Trail.stroke

# + tags=["hide_inp"]
help(Trail.stroke)
# -

# Trails can be turned to diagrams.

trail.stroke()

# ### Trail.rotate

# + tags=["hide_inp"]
help(Trail.rotate)
# -

# Trails can be transformed

trail2 = trail.rotate_by(0.2)
trail2.stroke()

# ### Trail.add

# + tags=["hide_inp"]
help(Trail.__add__)
# -

# Trails addition extends the trail.

(trail + trail2).stroke()
