# + tags=["hide"]
import math

from chalk.core import BaseDiagram
from chalk import *

def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
# -

# Diagrams can be rendered to SVG, PNG or PDF.

# ### Diagram.render

# + tags=["hide_inp"]
help(BaseDiagram.render)
# -

circle(1).render("circle.png")
from IPython.display import Image
Image("circle.png")
