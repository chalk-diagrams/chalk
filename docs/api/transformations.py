# + tags=["hide_inp"]
from chalk.core import BaseDiagram
from chalk import *

def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
# -


# Any Diagram (or other object in Chalk) can be transformed by affine transformation.
# These produce a new diagram in the standard manner.

# ### scale

# + tags=["hide_inp"]
help(BaseDiagram.scale)
# -

#

triangle(1) | triangle(1).scale(2)

# Transformations apply to the whole diagram.

(triangle(1) | triangle(1)).scale(2)


# ### translate

# + tags=["hide_inp"]
help(BaseDiagram.translate)
# -

#

triangle(1).translate(1, 1).show_envelope().show_origin()

#

triangle(1) + triangle(1).translate(1, 1)

# ### shear_x

# + tags=["hide_inp"]
help(BaseDiagram.shear_x)
# -

#

square(1).shear_x(0.25).show_envelope()

#

square(1) | square(1).shear_x(0.25)

# ### rotate

# + tags=["hide_inp"]
help(BaseDiagram.rotate)
# -

#

triangle(1) | triangle(1).rotate(90)

# ### rotate_by

# + tags=["hide_inp"]
help(BaseDiagram.rotate_by)
# -

#

triangle(1) | triangle(1).rotate_by(0.2)
