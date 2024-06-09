from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, Union

from colour import Color
from typing_extensions import Self

import chalk.transform as tx
from chalk.transform import ColorVec, Property, Scalars, Mask

PyCairoContext = Any
PyLatex = Any

PropLike = Union[Property, float]
ColorLike = Union[str, Color, ColorVec]


def to_color(c: ColorLike) -> ColorVec:
    if isinstance(c, str):
        return tx.np.array(Color(c).rgb)
    elif isinstance(c, Color):
        return tx.np.array(c.rgb)
    return c


FC = Color("white")
LC = Color("black")
LW = 0.1

STYLE_LOCATIONS = {
    "fill_color": (0, 3),
    "fill_opacity": (3, 4),
    "line_color": (4, 7),
    "line_opacity": (7, 8),
    "line_width": (8, 9),
    "output_size": (9, 10),
    "dashing": (10, 12),
}

DEFAULTS = {
    "fill_color": to_color(LC),
    "fill_opacity": tx.np.array(1.0),
    "line_color": to_color(LC),
    "line_opacity": tx.np.array(1.0),
    "line_width": tx.np.array(LW),
    "output_size": tx.np.array(200.0),
    "dashing": tx.np.array(0),
}
STYLE_SIZE = 12


class Stylable:
    def line_width(self, width: float) -> Self:
        return self.apply_style(Style(line_width_=width))

    # def line_width_local(self, width: float) -> Self:
    #     return self.apply_style(Style(line_width_=(WidthType.LOCAL, width)))

    def line_color(self, color: Color) -> Self:
        return self.apply_style(Style(line_color_=to_color(color)))

    def fill_color(self, color: Color) -> Self:
        return self.apply_style(Style(fill_color_=to_color(color)))

    def fill_opacity(self, opacity: float) -> Self:
        return self.apply_style(Style(fill_opacity_=opacity))

    def dashing(self, dashing_strokes: List[float], offset: float) -> Self:
        "TODO: implement this function."
        return self.apply_style(Style())

    def apply_style(self: Self, style: StyleHolder) -> Self:
        raise NotImplementedError("Abstract")


def m(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    return a if a is not None else b


class WidthType(Enum):
    LOCAL = auto()
    NORMALIZED = auto()


def Style(
    line_width_: Optional[PropLike] = None,
    line_color_: Optional[ColorLike] = None,
    line_opacity_: Optional[PropLike] = None,
    fill_color_: Optional[ColorLike] = None,
    fill_opacity_: Optional[PropLike] = None,
    dashing_: Optional[PropLike] = None,
    output_size: Optional[PropLike] = None,
) -> StyleHolder:
    b = (
        tx.np.zeros(STYLE_SIZE),
        tx.np.zeros(STYLE_SIZE, dtype=bool),
    )

    def update(b, key: str, value):  # type: ignore
        base, mask = b
        index = (Ellipsis, slice(*STYLE_LOCATIONS[key]))
        if value is not None:
            base = tx.index_update(base, index, value)  # type: ignore
            mask = tx.index_update(mask, index, True)  # type: ignore
        return base, mask

    b = update(b, "line_width", line_width_)
    b = update(b, "line_color", line_color_)
    b = update(b, "line_opacity", line_opacity_)
    b = update(b, "fill_color", fill_color_)
    b = update(b, "fill_opacity", fill_opacity_)
    b = update(b, "output_size", output_size)
    return StyleHolder(*b)


@dataclass
class StyleHolder(Stylable):
    """Style class."""

    base: Scalars
    mask: Mask

    def split(self, i: int) -> StyleHolder:
        return StyleHolder(base=self.base[i], mask=self.mask[i])

    def get(self, key: str) -> tx.Scalars:
        self.base = self.base
        v = self.base[slice(*STYLE_LOCATIONS[key])]
        return tx.np.where(
            self.mask[slice(*STYLE_LOCATIONS[key])], v, DEFAULTS[key]
        )

        # if self.mask[(1,) + key].all():
        #     return v
        # else:
        #     return None

    @property
    def line_width_(self) -> Property:
        return self.get("line_width")

    @property
    def line_color_(self) -> ColorVec:
        return self.get("line_color")

    @property
    def line_opacity_(self) -> Property:
        return self.get("line_opacity")

    @property
    def fill_color_(self) -> ColorVec:
        return self.get("fill_color")

    @property
    def fill_opacity_(self) -> Property:
        return self.get("fill_opacity")

    @property
    def output_size(self) -> Property:
        return self.get("output_size")

    @property
    def dashing_(self) -> None:
        return None

    @classmethod
    def empty(cls) -> StyleHolder:
        return cls(
            tx.np.zeros((STYLE_SIZE)),
            tx.np.zeros((STYLE_SIZE), dtype=bool),
        )

    @classmethod
    def root(cls, output_size: tx.Floating) -> StyleHolder:
        return Style(output_size=output_size)

    def apply_style(self, other: StyleHolder) -> StyleHolder:
        return self.merge(other)

    def merge(self, other: StyleHolder) -> StyleHolder:
        mask = self.mask | other.mask
        base = tx.np.where(other.mask, other.base, self.base)
        return StyleHolder(base, mask)

    def render(self, ctx: PyCairoContext) -> None:
        """Renders the style object.

        Args:
            ctx (PyCairoContext): A context.
        """
        if self.fill_color_ is not None:
            if self.fill_opacity_ is None:
                op = 1.0
            else:
                op = self.fill_opacity_[0]
            f = self.fill_color_
            ctx.set_source_rgba(f[0], f[1], f[2], op)
            ctx.fill_preserve()

        # set default values if they are not provided
        if self.line_color_ is None:
            lc = LC
        else:
            lc = self.line_color_
        # Set by observation
        assert self.output_size is not None
        normalizer = self.output_size * (15 / 500)
        if self.line_width_ is None:
            lw = LW * normalizer
        else:
            lw = self.line_width_
            # lwt, lw = self.line_width_
            # if lwt == WidthType.NORMALIZED:
            #     lw = lw * normalizer

            # elif lwt == WidthType.LOCAL:
            #     lw = lw
        ctx.set_source_rgb(*lc)
        ctx.set_line_width(lw.reshape(-1)[0])

        if self.dashing_ is not None:
            ctx.set_dash(self.dashing_[0], self.dashing_[1])

    def to_svg(self) -> str:
        """Converts to SVG.

        Returns:
            str: A string notation of the SVG.
        """
        style = ""
        if self.fill_color_ is not None:
            f = self.fill_color_ * 256
            style += f"fill: rgb({f[0]} {f[1]} {f[2]});"
        if self.line_color_ is not None:
            lc = self.line_color_ * 256
            style += f"stroke: rgb({lc[0]} {lc[1]} {lc[2]});"
        else:
            style += "stroke: black;"

        # Set by observation
        assert self.output_size is not None
        normalizer = self.output_size[0] * (15 / 500)
        if self.line_width_ is not None:
            lw = self.line_width_
            # if lwt == WidthType.NORMALIZED:
            #     lw = lw * normalizer
            # elif lwt == WidthType.LOCAL:
            #     lw = lw
        else:
            lw = LW * normalizer

        style += f" stroke-width: {lw.reshape(-1)[0]};"

        if self.fill_opacity_ is not None:
            style += f"fill-opacity: {self.fill_opacity_[0]};"
        if self.dashing_ is not None:
            style += (
                f"stroke-dasharray: {' '.join(map(str, self.dashing_[0]))};"
            )

        return style

    # def to_tikz(self, pylatex: PyLatex) -> Dict[str, str]:
    #     """Converts to dictionary of tikz options."""
    #     style = {}

    #     def tikz_color(color: Color) -> str:
    #         r, g, b = color.rgb
    #         return f"{{rgb,1:red,{r}; green,{g}; blue,{b}}}"

    #     if self.fill_color_ is not None:
    #         style["fill"] = tikz_color(self.fill_color_)
    #     if self.line_color_ is not None:
    #         style["draw"] = tikz_color(self.line_color_)
    #     # This constant was set based on observing TikZ output
    #     assert self.output_size is not None
    #     normalizer = self.output_size * (175 / 500)

    #     if self.line_width_ is not None:
    #         lwt, lw = self.line_width_
    #         if lwt == WidthType.NORMALIZED:
    #             lw = lw * normalizer
    #         elif lwt == WidthType.LOCAL:
    #             lw = lw
    #     else:
    #         lw = normalizer * LW
    #     style["line width"] = f"{lw}pt"
    #     if self.fill_opacity_ is not None:
    #         style["fill opacity"] = f"{self.fill_opacity_}"
    #     if self.dashing_ is not None:
    #         style["dash pattern"] = (
    #             f"{{on {self.dashing_[0][0]}pt off {self.dashing_[0][0]}pt}}"
    #         )

    #     return style
