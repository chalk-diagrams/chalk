# + tags=["hide_inp"]
from chalk.core import BaseDiagram
from chalk import *

def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
# -


# Diagrams can be styled using standard vector graphic style
# primitives. Colors use the Python [colour](https://github.com/vaab/colour) library.

from colour import Color
blue = Color("blue")
orange = Color("orange")

# ### Diagram.fill_color

# + tags=["hide_inp"]
help(BaseDiagram.fill_color)
# -

#

triangle(1).fill_color(blue)

# ### Diagram.fill_opacity

# + tags=["hide_inp"]
help(BaseDiagram.fill_opacity)
# -

#

triangle(1).fill_color(blue).fill_opacity(0.2)


# ### Diagram.line_color

# + tags=["hide_inp"]
help(BaseDiagram.line_color)
# -

#

triangle(1).line_color(blue)

# ### Diagram.line_width

# + tags=["hide_inp"]
help(BaseDiagram.line_width)
# -

#

triangle(1).line_width(0.05)


# ### Diagram.dashing

# + tags=["hide_inp"]
help(BaseDiagram.dashing)
# -

#

triangle(1).dashing([0.2, 0.1], 0) 


# ### Advanced Example


# Example: Outer styles override inner styles

(triangle(1).fill_color(orange) | square(2)).fill_color(blue)
