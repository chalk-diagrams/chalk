from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import PIL
from PIL import Image as Im

from chalk.shapes.shape import Shape
from chalk.transform import P2, BoundingBox, origin
from chalk.types import Diagram


def from_pil(
    im: Im,
    alpha: float = 1.0,
) -> Any:
    import cairo

    format: cairo.Format = cairo.FORMAT_ARGB32
    if "A" not in im.getbands():
        im.putalpha(int(alpha * 256.0))
    arr = bytearray(im.tobytes("raw", "BGRa"))
    surface = cairo.ImageSurface.create_for_data(
        arr, format, im.width, im.height  # type: ignore
    )
    return surface


@dataclass
class Image(Shape):
    """Image class."""

    local_path: str
    url_path: Optional[str]

    def __post_init__(self) -> None:
        if self.local_path.endswith("svg"):
            import cairosvg

            out = BytesIO()
            cairosvg.svg2png(url=self.local_path, write_to=out)
        else:
            out = open(self.local_path, "rb")  # type:ignore

        self.im = PIL.Image.open(out)
        self.height = self.im.height
        self.width = self.im.width

    def get_bounding_box(self) -> BoundingBox:
        left = origin.x - self.width / 2
        top = origin.y - self.height / 2
        tl = P2(left, top)
        br = P2(left + self.width, top + self.height)
        return BoundingBox([tl, br])


def image(local_path: str, url_path: Optional[str]) -> Diagram:
    from chalk.core import Primitive

    return Primitive.from_shape(Image(local_path, url_path))
