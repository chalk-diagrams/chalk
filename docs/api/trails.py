import sys
sys.path.append("/home/srush/Projects/diagrams/venv/lib/python3.9/site-packages")
import math
from chalk import *

# ### Trail


# Trails are a sequence of vectors.

trail = Trail([Vector(1, 0), Vector(1, 1), Vector(0, 1)])


# ### Trail.rotate

help(Trail.rotate)

# Trails can be transformed

trail2 = trail.rotate_by(0.2)


# ### Trail.__add__

help(Trail.__add__)

# Trails addition extends the trail.

trail = trail + trail2

# ### Trail.stroke

help(Trail.stroke)

# Trails can be turned to diagrams.

trail.stroke()

