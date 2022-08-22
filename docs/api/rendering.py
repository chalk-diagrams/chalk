# + tags=["hide"]
import math

from chalk.core import BaseDiagram
from chalk import *

def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
# -

# Chalk supports three back-ends (Cairo, SVG, TikZ),
# which allow the created `Diagram`s to be rendered as PNG, SVG, PDF files, respectively.
# The three corresponding methods for rendering are: `render`, `render_svg`, `render_pdf`;
# these are documented below.
#
# ### Diagram.render

# + tags=["hide_inp"]
help(BaseDiagram.render)
# -

circle(1).render("circle.png")
from IPython.display import Image
Image("circle.png")

# ### Diagram.render_svg

# + tags=["hide_inp"]
help(BaseDiagram.render_svg)
# -

# ### Diagram.render_pdf

# + tags=["hide_inp"]
help(BaseDiagram.render_pdf)
# -

# ### ``Diagram``s in IPython notebooks

# When a ``Diagram`` is used in an IPython notebook, it is automatically displayed as an SVG.
# To adjust the height of the generated image, one can use the `set_svg_height` function:

# + tags=["hide_inp"]
help(set_svg_height)
# -

# This function is particularly useful for showing tall drawings:

set_svg_height(500)
vcat([circle(1) for _ in range(5)])
