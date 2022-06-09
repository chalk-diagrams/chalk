import math
from chalk import *

# ### scale

help(Diagram.scale)

#

triangle(1) | triangle(1).scale(2)

# Transformations apply to the whole diagram.

(triangle(1) | triangle(1)).scale(2)


# ### translate

help(Diagram.translate)

#

triangle(1).translate(1, 1).show_bounding_box().show_origin()

#

triangle(1) + triangle(1).translate(1, 1)

# ### shear_x

help(Diagram.shear_x)

#

square(1).shear_x(0.25).show_bounding_box()

#

square(1) | square(1).shear_x(0.25)

# ### rotate

help(Diagram.rotate)

#

triangle(1) | triangle(1).rotate(math.pi)

# ### rotate_by

help(Diagram.rotate_by)

#

triangle(1) | triangle(1).rotate_by(0.2)


