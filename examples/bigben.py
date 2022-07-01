from chalk import *
from colour import Color
import math

gold = Color("gold")
white = Color("white")
black = Color("black")


# Outer frame

r = rectangle(1, 1).fill_color(gold)
inner = r.scale(0.9).fill_color(white)
outer = r + inner
outer


inner = circle(inner.get_envelope()(unit_x))
x = outer + inner

inner2 = inner.scale(0.5).get_envelope()

def rot_cycle(d: Diagram, times: int) -> Diagram:
    x = empty()
    for i in range(times):
        x += d.rotate_by(i / times)
    return x


tic = rectangle(1, 0.1).fill_color(black).align_r().translate(3, 0)

tics = rot_cycle(tic, 12).scale(1 / 3)

hand = (rectangle(1, 0.2) | triangle(0.7).rotate_by(1/4)).align_l().fill_color(black)
hand


watch = x + tics.scale_uniform_to_x(0.9) + hand.scale_uniform_to_x(0.9 / 2)
watch


corner = unit_x.rotate_by(1/4).arc()
corner

