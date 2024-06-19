import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

import os

import chalk.align as align
import chalk.core
import chalk.namespace
from chalk.align import *  # noqa: F403
from chalk.arrow import ArrowOpts, arrow_at, arrow_between, arrow_v
from chalk.combinators import *  # noqa: F403
from chalk.core import set_svg_draw_height, set_svg_height
from chalk.envelope import Envelope
from chalk.monoid import Maybe, MList, Monoid
from chalk.shapes import *  # noqa: F403
from chalk.style import Style, to_color
from chalk.subdiagram import Name
from chalk.trail import Trail
from chalk.transform import (
    P2,
    V2,
    Affine,
    BoundingBox,
    X,
    from_radians,
    to_radians,
)
from chalk.types import Diagram

unit_x = X.unit_x
unit_y = X.unit_y

if eval(os.environ.get("CHALK_JAX", "0")):
    import chex

    jax_type = [
        chalk.core.Primitive,
        chalk.core.Compose,
        chalk.core.Envelope,
        chalk.core.ApplyTransform,
        chalk.core.ComposeAxis,
        chalk.envelope.EnvDistance,
        chalk.style.StyleHolder,
        chalk.shapes.Trail,
        chalk.shapes.Path,
        chalk.shapes.Segment,
        chalk.trail.Located,
        chalk.shapes.Spacer,
    ]
    for t in jax_type:
        chex.register_dataclass_type_with_jax_tree_util(t)

if not TYPE_CHECKING:

    # Set library name the same as on PyPI
    # must be the same as setup.py:setup(name=?)
    __libname__: str = "chalk-diagrams"  # custom dunder attribute
    __version__: str = metadata.version(__libname__)
