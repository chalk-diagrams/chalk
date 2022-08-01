import math
from typing import Iterable, List, Optional, Tuple, Union

try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata  # type: ignore

import chalk.align as align
from chalk.align import *  # noqa: F403
from chalk.arrow import (
    ArrowOpts,
    arrow_at,
    arrow_between,
    arrow_v,
    make_path,
    unit_arc_between,
)
from chalk.combinators import *  # noqa: F403
from chalk.core import empty, set_svg_height
from chalk.envelope import Envelope
from chalk.image import image
from chalk.latex import latex
from chalk.path import Path
from chalk.shape import Arc, Circle, Rectangle, Spacer
from chalk.shapes import *  # noqa: F403
from chalk.style import Style
from chalk.text import text
from chalk.trail import Trail
from chalk.transform import (
    P2,
    V2,
    Affine,
    BoundingBox,
    from_radians,
    origin,
    to_radians,
    unit_x,
    unit_y,
)
from chalk.types import Diagram

# Set library name the same as on PyPI
# must be the same as setup.py:setup(name=?)
__libname__: str = "chalk-diagrams"  # custom dunder attribute
__version__ = metadata.version(__libname__)


ignore = [Trail, V2]
