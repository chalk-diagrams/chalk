from typing import Any, Optional, List, Tuple

from dataclasses import dataclass, fields

from colour import Color


PyCairoContext = Any


def m(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    return a if a is not None else b


LC = Color("black")
LW = 0.01


@dataclass
class Style:
    line_width: Optional[float] = None
    line_color: Optional[Color] = None
    fill_color: Optional[Color] = None
    fill_opacity: Optional[float] = None
    dashing: Optional[Tuple[List[float], float]] = None

    @classmethod
    def default(cls) -> "Style":
        return cls()

    def merge(self, other: "Style") -> "Style":
        return Style(
            *(
                m(getattr(other, dim.name), getattr(self, dim.name))
                for dim in fields(self)
            )
        )

    def render(self, ctx: PyCairoContext) -> None:
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
        style = ""
        if self.fill_color is not None:
            style += f"fill: {self.fill_color.hex_l};"
        if self.line_color is not None:
            style += f"stroke: {self.line_color.hex_l};"
        if self.line_width is not None:
            style += f"stroke-width: {self.line_width};"
        if self.fill_opacity is not None:
            style += f"fill-opacity: {self.fill_opacity};"
        if self.dashing is not None:
            style += (
                f"stroke-dasharray: {' '.join(map(str, self.dashing[0]))};"
            )

        return style
