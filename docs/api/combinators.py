import math
from chalk import *

# ### above

help(above)

#

diagram = triangle(1) / square(1)
diagram

#

diagram.show_bounding_box().show_origin()


# ### atop

help(atop)

# Example 1 - Atop at origin

diagram = square(1) + triangle(1)
diagram

#

diagram.show_bounding_box().show_origin()

# Example 2 - Align then atop origin

s = square(1).align_r().align_b().show_origin()
t = triangle(1).align_l().align_t().show_origin()
s

#

t

#

s + t

# ### beside

help(beside)

#

diagram = triangle(1) | square(1)
diagram

#

triangle(1).show_origin() | square(1).show_origin()

#

diagram.show_origin()


# ### vcat

help(vcat)

#

vcat([triangle(1), square(1), triangle(1)], 0.2)

# ### concat

help(concat)

#

concat([triangle(1), square(1), triangle(1)])


# ### hcat

help(hcat)

#

hcat([triangle(1), square(1), triangle(1)], 0.2)
