# # Big Ben

# This notebook is an annotated tutorial on how to use all the features of Chalk
# for creating intricate diagrams. Specifically we are going to recreate the clockface of Big Ben.


# !<img style="height:500px;" src="../bigben.png">


# Here is what it will look like when we are done.

# !<img style="height:500px;" src="../complete.svg">

from chalk import *
from colour import Color
import chalk

gold = Color("#E7D49C")
white = Color("white")
black = Color("black")
grey = Color("grey")

# ## Preliminary: Roman Numerals


# ![](../part0.png)

# We start by drawing some basic diagrams. Here is a rectangular column.
# Each diagram has style properties and an "envelope" that describes its
# boundaries. Envelopes are a bit complex, they roughly describe the boundaries
# of the diagram.

column = rectangle(1, 4).fill_color(black)
column.show_envelope()


# The benefit of the envelope representation (over something simpler like
# a bounding box) is that it behaves more intuitively under affine transformations
# like rotation.

diamond = rectangle(1, 1).fill_color(black).rotate_by(1 / 8)
diamond.show_envelope()


# Envelopes are used for combining diagrams. We can easily stack our shapes
# using the `beside` combinator.

column.show_beside(diamond, -unit_y)


# We can also change the envelope of diagrams to help with object placement.
# For instance here we substitute a small envelope.

column = column.with_envelope(rectangle(1, 2.5))
column.show_beside(diamond, -unit_y)

# When stacking on top we can use the shortcut `/`.

i = (diamond / column / diamond).center_xy()
i.show_envelope()

# We can also put diagrams next to each other using the `|` combinator.

ii = i | i
ii

# We can similar create diagrams for the other roman numerals.
# The `align` function changes the center (red dot) of a diagram. The `+`
# combinator attaches diagrams at the origin.

v = i.align_b() + rectangle(1.5, 1).fill_color(black).align_bl()
v

v = (v.align_br() + i.align_b()).center_xy()
v


# Creating X is a bit harder. We use two transformations to help us out.
# Using `translate` helps us nudge the center alignment of the diamonds over.
# Using `shear` lets us create a center line with a diagonal slash.

ddiamond = (diamond | diamond).translate(-0.5, 0)
ddiamond.show_origin()

x = ddiamond / column / ddiamond
x = (x.center_xy() + rectangle(2, 0.5).fill_color(black).shear_y(-0.2)).center_xy()
x


# Now we have all of our roman numerals, we can list them in clock-wise order.

numbers = [
    x | i | i,
    i,
    i | i,
    i | i | i,
    i | v,
    v,
    v | i,
    v | i | i,
    v | i | i | i,
    i | x,
    x,
    x | i,
]


# We can draw the main clock-face by moving each number to the edge, and then rotating it
# to its location. The `concat` combinator glues each of these together at the origin.

chalk.set_svg_height(500)
part0 = concat(
    [
        n.center_xy().scale(0.05).translate_by(-unit_y).rotate_by(-i / 12)
        for i, n in enumerate(numbers)
    ]
)

part0.show_origin()


# ## Inner Pattern

# ![](../part1.png)


# In order to draw the inner part of the clock, we are going to need diagrams
# that are rotationally symmetric around 360 degrees. To do this we will
# introduce a function that takes a diagram and repeats it with rotation.


def rot_cycle(d: Diagram, times: int) -> Diagram:
    return concat(d.rotate_by(i / times) for i in range(times))


width = -4.4
inner_circle = rot_cycle(circle(1.1).translate(0, width), 12).rotate_by(
    (1 / 12) / 2
) + circle(3).fill_color(black)
inner_circle


# Now we want to trace a trail that looks like the inner patttern.
# I do a little geometry here to estimate the shapes.
# Vectors `unit_y` and `unit_x` are geometric helpers.

# It looks like
# one is 45 degress and the other is 60 degress. We can calulate
# lenths to make this work.

u45 = unit_x.rotate(-45)
u60 = unit_x.rotate(60)
diffy = abs(u45.y / u60.y)
diffx = diffy * abs(u60.x / u45.x)

# A `Trail` is a sequence of vectors drawn in order.
# Once you are done drawing one you can use `stroke` to
# make it a diagram.

fudge = 0.73

y = (
    Trail(
        [
            u45,
            diffy * u60,
            -diffy * u60,
            -fudge * u60,
            fudge * u60,
            diffx * u45,
            diffx * unit_y,
        ]
    )
    .stroke()
    .align_br()
)
y


# Below we draw a curve. The `arc_between` function makes
# it easy to draw a curve between two points with a specified radius.

curve = 0.5
under_arc = arc_between(-unit_x, 2 * -unit_y, curve).align_tr()
under_arc.show_origin()

# We then combine them as reflect to make make the whole pattern.

pattern = (y.scale(3) / under_arc).align_r()
pattern = (pattern + pattern.reflect_x()).align_b()
pattern


# We can then rotate this to create the whole pattern. We set the
# fudge factor above to make the pattern connect.

part1 = inner_circle + rot_cycle(pattern.translate(0, width - 1), 12)
part1 = part1.line_color(gold).line_width(0.03)
part1

# Looks pretty close to the original!

# ![](../part1.png)


# ## Outer Bands


# ![](../part2.png)


# The first band is two circles with a black dots.

dots = rectangle(0.05, 0.05).fill_color(black)
band1 = (circle(1.1) + circle(1)).line_width(0.03) + rot_cycle(
    dots.translate(1.05, 0), 12
)
band1


# Band two has thin dividing lines and the hour markers from part 0.
# In order to fit in the numbers we write a function that lets us scale to a given circle.


def fit_in(b: Diagram, s: Diagram, frame=0.1) -> Diagram:
    # Find the inner radius
    m = min([x for x in b.get_trace()(origin, unit_x) if x > 0])

    # Scale the inner diagram to that size
    return b + s.scale_uniform_to_x(2 * m - frame)


# Make the thin lines and the numbers.


# lines = make_path([origin, 0.4 * unit_x]).center_xy().line_width(0.001)
lines = rectangle(0.4, 0.001).fill_color(black).center_xy().line_width(0.001)
band2 = (
    fit_in(circle(1.4), part0, 0.1) + circle(1) + rot_cycle(lines.translate(1.2, 0), 48)
)
band2


# The third band has a little jewel cross.

diamond = rectangle(1, 1).rotate_by(1 / 8)
s = (
    (diamond | rectangle(2, 1).with_envelope(rectangle(1, 1)) | diamond)
    .fill_color(black)
    .center_xy()
)
s = s.scale_y(0.5) + s.rotate_by(0.25).scale_y(0.75).scale_x(0.25)
jewel = s.rotate_by(0.25)
jewel

lines2 = rectangle(0.3, 0.001).fill_color(black).center_xy().line_width(0.03)
lines = rectangle(0.6, 0.001).fill_color(black).center_xy().line_width(0.001)

# Draw the outlines

a = 1.8
c = 1.7
b = 1.6
band3 = (
    circle(2.0).line_width(0.1)
    + (circle(a) + circle(b)).line_width(0.05)
    + circle(1.4).line_width(0.08)
)
band3

# Add the lines

band3 = (
    band3
    + rot_cycle(lines.translate(c, 0), 60)
    + rot_cycle(lines2.translate(c, 0).rotate_by(1 / 48), 48)
)
band3

# Add the jewel

band3 = band3 + rot_cycle(
    jewel.center_xy().scale_uniform_to_x((a - b) * 2).translate(0, c), 12
)
band3


part2 = fit_in(band3, fit_in(band2, band1))
part2

# Looks pretty close to the original!

# ![](../part2.png)

# ## Frame

# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Big_Ben_Clock_Face.jpg/612px-Big_Ben_Clock_Face.jpg?20140924183502)


# The whole clock is surrounded by a thick gold frame.

r = rectangle(1, 1).fill_color(black).line_color(gold).line_width(0.1)
r

# Next we do each of the outer corners by themselves. We first make a sloping triangle shape.

corner = (
    make_path([origin, unit_x]).align_l()
    + make_path([origin, -unit_y])
    + arc_between((0, -1), (1, 0), -0.2)
).line_width(0.03)
corner


# Internally there is a little golden pattern. To make this we use the `juxtapose` function
# which moves a diagram to be next to another along an angle.

decal = rectangle(2, 2).with_envelope(rectangle(0, 0)).fill_color(gold)
c = circle(1).fill_color(black)
decal = concat(
    [
        decal.juxtapose(c, unit_y),
        decal.juxtapose(c, -unit_y),
        decal.juxtapose(c, unit_x),
        decal.juxtapose(c, -unit_x),
        decal,
    ]
)

decal = decal.line_color(gold).line_width(0.03)
outer = decal.get_envelope().width / 2
decal = circle(outer).fill_color(black) + decal
decal

# To add the other shapes with we `arc` which draws part of a circle.

marc = arc(outer / 2, 90, 3.5 * 90)
decal = concat(
    [
        decal,
        decal.juxtapose(marc.align_t(), unit_x),
        decal.juxtapose(marc.rotate_by(1 / 2).align_r(), -unit_y),
    ]
)
decal = decal.line_width(0.03)
decal


# We scale the decoration to fit in the corner we created

fudge = 0.57
corner = corner.align_bl() + decal.scale_uniform_to_x(fudge).align_bl()
corner = corner.align_tr().translate(-0.4, 0.4).line_color(gold)
corner

# Then we add a black circle in the center.

inner = (
    circle(1)
    .line_width(0.03)
    .line_color(gold)
    .fill_color(black)
    .scale_uniform_to_x(1 - 0.04)
)
part3 = fit_in(r, rot_cycle(corner, 4), 0.05) + inner
part3


# ## Clock Hands

# ![](../part1.png)


# Trace a shape of half a clock hand.

hand = make_path(
    [(2, -0.5), (1, -0), (0.4, 20), (0, 21), (0, -1.5), (0.5, -1), (2, -0.5)]
).fill_color(black)
hand = (hand + hand.reflect_x()).translate(0, -4).line_width(0.01)
hand.show_origin()

hand2 = make_path(
    [
        (1, 0),
        (1, 7),
        (2, 7),
        (2.5, 8),
        (2, 8.5),
        (1, 9.5),
        (0.3, 11),
        (0, 12),
        (0, 0),
        (1, 0),
    ]
).fill_color(black)
hand2 = (hand2 + hand2.reflect_x()).translate(0, -3).line_width(0.01).line_color(grey)
hand2.show_origin()


part4 = hand2.scale_uniform_to_y(0.5).rotate_by(0.07) + hand.scale_uniform_to_y(1.0)
part4


# ## All Together

part2

final = (
    part3
    + fit_in(inner, (fit_in(part2, part1)), 0.0)
    + part4.scale_x(0.8).scale(0.55).rotate_by(0.10)
)

final.render_svg("complete.svg", height=500)


# !<img style="height:500px;" src="../complete.svg">


# !<img style="height:500px;" src="../bigben.png">
