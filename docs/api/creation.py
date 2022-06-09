import math
from chalk import *
def help(f):
    import pydoc
    from IPython.display import HTML
    return HTML(pydoc.HTMLDoc().docroutine(f))
    

# ### circle

help(circle)

# ::: chalk.circle


circle(1) + circle(0.5)

# ### arc

help(arc)

#

quarter = math.pi / 2
arc(1, 0, quarter) 

#

arc(1, 0, quarter) + arc(1, 2 * quarter, 3 * quarter) 


# ### arc_between

help(arc_between)

#

arc_between((0, 0), (1, 0), 1)

# ### polygon

help(polygon)

#

polygon(8, 2)


# ### square

help(square)

#

square(1)


# ### triangle

help(triangle)

#

triangle(1)


# ### rectangle

help(rectangle)

#

rectangle(8, 2, 0.5)



# ### make_path

help(make_path)

#

make_path([(0, 0), (0, 1), (1, 1), (1, 2)])

# ### text

help(text)

#

text("hello", 0.2)

#

text("hello", 0.2).show_bounding_box()
