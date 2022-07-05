from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

from colour import Color

PyCairoContext = Any
PyLatex = Any


def m(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    return a if a is not None else b


LC = Color("black")
LW = 0.01


@dataclass
class Style:
    """Style class."""

    line_width: Optional[float] = None
    line_color: Optional[Color] = None
    fill_color: Optional[Color] = None
    fill_opacity: Optional[float] = None
    dashing: Optional[Tuple[List[float], float]] = None
    scale: Optional[float] = 1.0

    @classmethod
    def default(cls) -> "Style":
        return cls()

    def scale_style(self, s: float) -> "Style":
        return Style(
            self.line_width,
            self.line_color,
            self.fill_color,
            self.fill_opacity,
            self.dashing,
            (1 if not self.scale else self.scale) * abs(s),
        )

    def merge(self, other: "Style") -> "Style":
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
        if self.fill_color:
            ctx.set_source_rgb(*self.fill_color.rgb)
            ctx.fill_preserve()

        # set default values if they are not provided
        if self.line_color is None:
            lc = LC
        else:
            lc = self.line_color

        if self.line_width is None:
            lw = LW
        else:
            lw = self.line_width

        ctx.set_source_rgb(*lc.rgb)
        ctx.set_line_width(lw)

        if self.dashing is not None:
            ctx.set_dash(self.dashing[0], self.dashing[1])

        ctx.stroke()

    def to_svg(self) -> str:
        """Converts to SVG.

        Returns:
            str: A string notation of the SVG.
        """
        style = ""
        if self.fill_color is not None:
            style += f"fill: {self.fill_color.hex_l};"
        if self.line_color is not None:
            style += f"stroke: {self.line_color.hex_l};"
        if self.line_width is not None:
            style += f"stroke-width: {15 * self.line_width};"
        if self.fill_opacity is not None:
            style += f"fill-opacity: {self.fill_opacity};"
        if self.dashing is not None:
            style += (
                f"stroke-dasharray: {' '.join(map(str, self.dashing[0]))};"
            )

        return style

    def to_tikz(self, pylatex: PyLatex) -> Dict[str, str]:
        """Converts to dictionary of tikz options."""
        style = {}

        def tikz_color(color: Color) -> str:
            r, g, b = color.rgb
            return f"{{rgb,1:red,{r}; green,{g}; blue,{b}}}"

        if self.fill_color is not None:
            style["fill"] = tikz_color(self.fill_color)
        if self.line_color is not None:
            style["draw"] = tikz_color(self.line_color)
        # This constant was set based on observing TikZ output
        width_scale = 10  # 20 * (1 if not self.scale else self.scale)
        if self.line_width is not None:
            style["line width"] = f"{width_scale * self.line_width}pt"
        else:
            style["line width"] = f"{width_scale * LW}pt"
        if self.fill_opacity is not None:
            style["fill opacity"] = f"{self.fill_opacity}"

        return style
