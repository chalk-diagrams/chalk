from colour import Color
from chalk import *
from chalk.arrow import connect_outside_elbow

color = Color("pink")

def make_dia():
    c1 = circle(0.75).fill_color(color).named("src") + text("src", 0.7)
    c2 = circle(0.75).fill_color(color).named("tgt") + text("tgt", 0.7)
    return c1 + c2.translate(3, 3)

dia1 = make_dia()
dia1 = connect_outside_elbow(dia1, "src", "tgt", "hv")

dia2 = make_dia()
dia2 = connect_outside_elbow(dia2, "src", "tgt", "vh")

dia = hcat([dia1, dia2], sep=2)

path = "examples/output/connect_elbow.svg"
dia.render_svg(path, height=256)
