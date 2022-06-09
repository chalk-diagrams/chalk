import math
from chalk import *


# ### Diagram.named

help(Diagram.named)

#

diagram = triangle(1).named("x") | square(1)
diagram

# ### place_on_path

help(place_on_path)

#

place_on_path([triangle(0.4) for i in range(5)],
              Path.from_list_of_tuples([(i, i % 2) for i in range(5)]))

# ### Diagram.get_subdiagram_bounding_box

help(Diagram.get_subdiagram_bounding_box)

#

diagram = triangle(1).named("x") | square(1) 
bbox = diagram.get_subdiagram_bounding_box("x")
diagram + place_on_path([triangle(0.4)], Path.from_point(bbox.center))

# ### connect

help(connect)

#

diagram = triangle(1).named("x") / square(1).named("y")
diagram + connect(diagram, "x", "y")


# ### connect_outer

help(connect_outer)

#

diagram = square(1).named("x") | hstrut(1) | square(1).named("y")
diagram + connect_outer(diagram, "x", "E", "y", "W")
