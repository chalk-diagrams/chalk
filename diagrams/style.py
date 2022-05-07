from typing import Any, Optional, List, Tuple

from dataclasses import astuple, dataclass, fields


from colour import Color  # type: ignore


PyCairoContext = Any

def m(a, b):
    return a if a is not None else b

LC = Color("black")
LW = 0.01

@dataclass
class Style:
    line_width: Optional[float] = None
    line_color: Optional[Color] = None
    fill_color: Optional[Color] = None
    dashing: Optional[Tuple[List[float], float]] = None

    @classmethod
    def default(cls) -> "Style":
        return cls(line_width=LW, line_color=LC)

    def merge(self, other: "Style"):
        return Style(*(m(getattr(other, dim.name), getattr(self, dim.name))
                       for dim in fields(self)))        
        
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
