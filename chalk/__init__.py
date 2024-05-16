import math
import sys
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Union

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

import chalk.align as align
from chalk.align import *  # noqa: F403
from chalk.arrow import ArrowOpts, arrow_at, arrow_between, arrow_v
from chalk.combinators import *  # noqa: F403
from chalk.core import set_svg_draw_height, set_svg_height
from chalk.envelope import Envelope
from chalk.monoid import Maybe, MList, Monoid
from chalk.shapes import *  # noqa: F403
from chalk.style import Style
from chalk.subdiagram import Name
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

if not TYPE_CHECKING:

    # Set library name the same as on PyPI
    # must be the same as setup.py:setup(name=?)
    __libname__: str = "chalk-diagrams"  # custom dunder attribute
    __version__: str = metadata.version(__libname__)
