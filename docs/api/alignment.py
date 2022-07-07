# + tags=["hide_inp"]
from chalk import *
def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
# -


# Each diagram has an associated bounding box and origin.
# Manipulating the position of the diagram within its bounding
# box allows for precise control of layout. Note that the
# Chalk API is immutable and always returns a new Diagram object.

# ### Diagram.show_origin

# + tags=["hide_inp"]
help(Diagram.show_origin)
# -

#

triangle(1).show_origin()

# ### Diagram.show_envelope

# + tags=["hide_inp"]
help(Diagram.show_envelope)
# -

rectangle(1, 1).show_envelope()


triangle(1).show_envelope()


rectangle(1, 1).show_beside(triangle(1), unit_x)


(rectangle(1, 1) | triangle(1)).pad(1.4)


arc(1, 0, math.pi).show_origin().show_envelope(rate=10)



# ### Diagram.align_*

# + tags=["hide_inp"]
help(Diagram.align_t)
# -

#

triangle(1).align_t().show_envelope()



triangle(1).align_t().show_beside(rectangle(1, 1).align_b(), unit_x)



# + tags=["hide_inp"]
help(Diagram.align_r)
# -

#

triangle(1).align_r().show_envelope().show_origin()

# ### Diagram.center_xy

# + tags=["hide_inp"]
help(Diagram.center_xy)
# -

#

triangle(1).center_xy().show_envelope().show_origin()


# ### Diagram.pad_*

# + tags=["hide_inp"]
help(Diagram.pad)
# -


#

triangle(1).pad(1.5).show_envelope().show_origin()


# ### Diagram.with_envelope

# + tags=["hide_inp"]
help(Diagram.with_envelope)
# -


#

from colour import Color

(rectangle(1, 1) + triangle(0.5)) | rectangle(1, 1)


(rectangle(1, 1) + triangle(0.5)).with_envelope(triangle(0.5)) | rectangle(1, 1).fill_color(Color("red"))
