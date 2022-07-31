from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import PIL
from svgwrite import Drawing
from svgwrite.base import BaseElement

from chalk.shape import Shape
from chalk.style import Style
from chalk.transform import P2, BoundingBox, origin
from chalk.types import Diagram, PyCairoContext, PyCairoSurface


def from_pil(
    im: PIL.Image,
    alpha: float = 1.0,
) -> PyCairoSurface:
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

    def render(self, ctx: PyCairoContext, style: Style) -> None:
        surface = from_pil(self.im)
        ctx.set_source_surface(surface, -(self.width / 2), -(self.height / 2))
        ctx.paint()

    def render_svg(self, dwg: Drawing, style: Style) -> BaseElement:
        dx = -self.width / 2
        dy = -self.height / 2
        return dwg.image(
            href=self.url_path, transform=f"translate({dx}, {dy})"
        )


def image(local_path: str, url_path: Optional[str]) -> Diagram:
    from chalk.core import Primitive

    return Primitive.from_shape(Image(local_path, url_path))
