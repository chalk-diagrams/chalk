from typing import Any, Optional

from dataclasses import dataclass

from colour import Color  # type: ignore


PyCairoContext = Any


@dataclass
class Style:
    LC = Color("black")
    LW = 0.01

    def __init__(
        self,
        line_width: Optional[float] = None,
        line_color: Optional[Color] = None,
        fill_color: Optional[Color] = None,
    ):
        self.line_width = line_width
        self.line_color = line_color
        self.fill_color = fill_color

    @classmethod
    def default(cls) -> "Style":
        return cls(line_width=cls.LW, line_color=cls.LC)

    def render(self, ctx: PyCairoContext) -> None:
        if self.fill_color:
            ctx.set_source_rgb(*self.fill_color.rgb)
            ctx.fill_preserve()

        # set default values if they are not provided
        lc = self.line_color or self.LC
        lw = self.line_width or self.LW

        ctx.set_source_rgb(*lc.rgb)
        ctx.set_line_width(lw)
        ctx.stroke()
