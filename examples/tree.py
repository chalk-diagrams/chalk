from PIL import Image as PILImage
from chalk import *
from chalk.transform import *
from chalk.trail import unit_x, unit_y
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
    
def draw_tree(tree, name="", ysep=8, xsep=0):
    node = circle(1).named(name)
    if tree is None:
        node = node.fill_color(blue)
        return name, node
    node = node.fill_color(blue if tree[0] else red)

    # Draw subtrees.
    lname, l = draw_tree(tree[1], name + "l")
    rname, r = draw_tree(tree[2], name + "r")

    # Position node an origin and subtrees to both sides.
    off = (l.get_bounding_box().right - r.get_bounding_box().left + xsep) / 2    
    x = node / vstrut(ysep) /  (l | hstrut(xsep) | r).translate(-off, 0)

    # Connect to children
    x = connect(x, name, lname) + connect(x, name, rname) + x
    return name, x


_, d =  draw_tree((True, (True, None, None), (False, (True, None, None), None)))
d.line_width(0.05)

_, d = draw_tree(sample_tree())
d = d.line_width(0.05)

d.render_svg("examples/output/tree.svg", height=500)
d.render("examples/output/tree.png", height=500)
PILImage.open("examples/output/tree.png")

