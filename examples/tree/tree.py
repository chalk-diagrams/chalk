from PIL import Image as PILImage
from chalk import *
from chalk.transform import *
import random

from chalk import transform as tx
from colour import Color

blue = Color("blue")
red = Color("red")

random.seed(10)
def flip(p=0.4):
    return random.random() > p

def sample_tree(n=1):
    "Sample a most right-branching tree"
    if n > 20:
        return None
    else:
        return (flip(),
                sample_tree(n+1) if flip(0.55)  else None,
                sample_tree(n+1) if flip(0.2) else None)


def draw_tree(tree, name="", ysep=5, xsep=1):
    node = circle(5).named(name)
    if tree is None:
        node = node.fill_color(blue)
        return name, node
    node = node.fill_color(blue if tree[0] else red)

    # Draw subtrees.
    lname, l = draw_tree(tree[1], name + "l")
    rname, r = draw_tree(tree[2], name + "r")

    # Position node an origin and subtrees to both sides.
    off = (l.get_envelope()(unit_x) + r.get_envelope()(-unit_x) + xsep) / 2    
    x = node / vstrut(ysep) /  (l | hstrut(xsep) | r).translate(-off, 0)

    # Connect to children
    x = x.connect_outside(name, lname, ArrowOpts(head_arrow=empty()))
    x = x.connect_outside(name, rname, ArrowOpts(head_arrow=empty()))
    return name, x


_, d =  draw_tree((True, (True, None, None), (False, (True, None, None), None)))
d.line_width(0.01)

_, d = draw_tree(sample_tree())
d = d.line_width(0.01)

d.render_svg("examples/output/tree.svg", 1200)
d.render("examples/output/tree.png", 1200)
PILImage.open("examples/output/tree.png")
