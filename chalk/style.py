from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar

from colour import Color

PyCairoContext = Any
PyLatex = Any


TStyle = TypeVar("TStyle", bound="StylableProtocol")


class StylableProtocol(Protocol):
    def line_width(self: TStyle, width: float) -> TStyle:
        ...

    def line_width_local(self: TStyle, width: float) -> TStyle:
        ...

    def line_color(self: TStyle, color: Color) -> TStyle:
        ...

    def fill_color(self: TStyle, color: Color) -> TStyle:
        ...

    def fill_opacity(self: TStyle, opacity: float) -> TStyle:
        ...

    def dashing(
        self: TStyle, dashing_strokes: List[float], offset: float
    ) -> TStyle:
        ...

    def apply_style(self: TStyle, style: Style) -> TStyle:
        ...


class Stylable:
    def line_width(self: TStyle, width: float) -> TStyle:
        return self.apply_style(
            Style(line_width_=(WidthType.NORMALIZED, width))
        )

    def line_width_local(self: TStyle, width: float) -> TStyle:
        return self.apply_style(Style(line_width_=(WidthType.LOCAL, width)))

    def line_color(self: TStyle, color: Color) -> TStyle:
        return self.apply_style(Style(line_color_=color))

    def fill_color(self: TStyle, color: Color) -> TStyle:
        return self.apply_style(Style(fill_color_=color))

    def fill_opacity(self: TStyle, opacity: float) -> TStyle:
        return self.apply_style(Style(fill_opacity_=opacity))

    def dashing(
        self: TStyle, dashing_strokes: List[float], offset: float
    ) -> TStyle:
        return self.apply_style(Style(dashing_=(dashing_strokes, offset)))


def m(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    return a if a is not None else b


class WidthType(Enum):
    LOCAL = auto()
    NORMALIZED = auto()


LC = Color("black")
LW = 0.1


@dataclass
class Style(Stylable):
    """Style class."""

    line_width_: Optional[Tuple[WidthType, float]] = None
    line_color_: Optional[Color] = None
    fill_color_: Optional[Color] = None
    fill_opacity_: Optional[float] = None
    dashing_: Optional[Tuple[List[float], float]] = None
    output_size: Optional[float] = None

    @classmethod
    def empty(cls) -> Style:
        return cls()

    @classmethod
    def root(cls, output_size: float) -> Style:
        return cls(output_size=output_size)

    def apply_style(self, other: Style) -> Style:
        return self.merge(other)

    def merge(self, other: Style) -> Style:
        """Merges two styles and returns the merged style.

        Args:
            other (Style): Another style object.

        Returns:
            Style: A style object.
        """
        return Style(
            *(
                m(getattr(other, dim.name), getattr(self, dim.name))
                for dim in fields(self)
            )
        )

    def render(self, ctx: PyCairoContext) -> None:
        """Renders the style object.

        Args:
            ctx (PyCairoContext): A context.
        """
        if self.fill_color_:
            if self.fill_opacity_ is None:
                op = 1.0
            else:
                op = self.fill_opacity_

            ctx.set_source_rgba(*self.fill_color_.rgb, op)
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
            lwt, lw = self.line_width_
            if lwt == WidthType.NORMALIZED:
                lw = lw * normalizer

            elif lwt == WidthType.LOCAL:
                lw = lw
        ctx.set_source_rgb(*lc.rgb)
        ctx.set_line_width(lw)

        if self.dashing_ is not None:
            ctx.set_dash(self.dashing_[0], self.dashing_[1])

    def to_svg(self) -> str:
        """Converts to SVG.

        Returns:
            str: A string notation of the SVG.
        """

        style = ""
        if self.fill_color_ is not None:
            style += f"fill: {self.fill_color_.hex_l};"
        if self.line_color_ is not None:
            style += f"stroke: {self.line_color_.hex_l};"
        else:
            style += "stroke: black;"

        # Set by observation
        assert self.output_size is not None
        normalizer = self.output_size * (15 / 500)
        if self.line_width_ is not None:
            lwt, lw = self.line_width_
            if lwt == WidthType.NORMALIZED:
                lw = lw * normalizer
            elif lwt == WidthType.LOCAL:
                lw = lw
        else:
            lw = LW * normalizer

        style += f"stroke-width: {lw};"

        if self.fill_opacity_ is not None:
            style += f"fill-opacity: {self.fill_opacity_};"
        if self.dashing_ is not None:
            style += (
                f"stroke-dasharray: {' '.join(map(str, self.dashing_[0]))};"
            )

        return style

    def to_tikz(self, pylatex: PyLatex) -> Dict[str, str]:
        """Converts to dictionary of tikz options."""
        style = {}

        def tikz_color(color: Color) -> str:
            r, g, b = color.rgb
            return f"{{rgb,1:red,{r}; green,{g}; blue,{b}}}"

        if self.fill_color_ is not None:
            style["fill"] = tikz_color(self.fill_color_)
        if self.line_color_ is not None:
            style["draw"] = tikz_color(self.line_color_)
        # This constant was set based on observing TikZ output
        assert self.output_size is not None
        normalizer = self.output_size * (175 / 500)
        if self.line_width_ is not None:
            lwt, lw = self.line_width_
            if lwt == WidthType.NORMALIZED:
                lw = lw * normalizer
            elif lwt == WidthType.LOCAL:
                lw = lw
        else:
            lw = normalizer * LW
        style["line width"] = f"{lw}pt"
        if self.fill_opacity_ is not None:
            style["fill opacity"] = f"{self.fill_opacity_}"

        return style
