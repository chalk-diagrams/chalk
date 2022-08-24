# + tags=["hide_inp"]
from chalk.core import BaseDiagram
from chalk import *

def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
# -


# Each diagram has an origin and an envelope.
# Manipulating the position of the diagram with respect to its origin and envelope allows for precise control of the layout.
# Note that the Chalk API is immutable and always returns a new ``Diagram`` object.

# ### Diagram.show_origin

# + tags=["hide_inp"]
help(BaseDiagram.show_origin)
# -

#

triangle(1).show_origin()

# ### Diagram.show_envelope

# + tags=["hide_inp"]
help(BaseDiagram.show_envelope)
# -

rectangle(1, 1).show_envelope()


triangle(1).show_envelope()


rectangle(1, 1).show_beside(triangle(1), unit_x)


(rectangle(1, 1) | triangle(1)).pad(1.4)


arc(1, 0, math.pi).show_origin().show_envelope(angle=10)

# ### Diagram.align_*

# + tags=["hide_inp"]
help(BaseDiagram.align_t)
# -

#

triangle(1).align_t().show_envelope()


triangle(1).align_t().show_beside(rectangle(1, 1).align_b(), unit_x)


# + tags=["hide_inp"]
help(BaseDiagram.align_r)
# -

#

triangle(1).align_r().show_envelope().show_origin()

# ### Diagram.center_xy

# + tags=["hide_inp"]
help(BaseDiagram.center_xy)
# -

#

triangle(1).center_xy().show_envelope().show_origin()


# ### Diagram.pad_*

# + tags=["hide_inp"]
help(BaseDiagram.pad)
# -


#

triangle(1).pad(1.5).show_envelope().show_origin()


# ### Diagram.with_envelope

# + tags=["hide_inp"]
help(BaseDiagram.with_envelope)
# -


#

from colour import Color

(rectangle(1, 1) + triangle(0.5)) | rectangle(1, 1)


(rectangle(1, 1) + triangle(0.5)).with_envelope(triangle(0.5)) | rectangle(1, 1).fill_color(Color("red"))
