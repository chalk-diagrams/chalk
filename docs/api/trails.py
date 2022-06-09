import math
from chalk import *
def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))

# ### Trail


# Trails are a sequence of vectors.

trail = Trail([Vector(1, 0), Vector(1, 1), Vector(0, 1)])


# ### Trail.rotate

help(Trail.rotate)

# Trails can be transformed

trail2 = trail.rotate_by(0.2)


# ### Trail.__add__

help(Trail.__add__)

# Trails addition extends the trail.

trail = trail + trail2

# ### Trail.stroke

help(Trail.stroke)

# Trails can be turned to diagrams.

trail.stroke()

