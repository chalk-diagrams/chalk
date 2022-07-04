# # Big Ben

# Big Ben is perhaps the most iconic clock face with the world standing at 22.5 feet.

# !<img style="height:500px;width:500px;max-width:auto;" src="bigben.png">

# In this notebook, we are going to replicate the design of the clockface from first principles using the
# Chalk library. This project was done for fun without any knowledge of clockmaking or even the right
# terminology. It is meant mainly as an introduction programmatic 2D diagramming.

# Here is what it will look like when we are done.

# !<img style="height:500px;width:500px;max-width:auto;" src="complete.svg">

# ## Preliminary: Roman Numerals

from chalk import *
from colour import Color

# The whole diagram a simple color pallet of gold black and a bit of grey. 

gold = Color("#E7D49C")
white = Color("white")
black = Color("black")
grey = Color("grey")


# To begin, we will introduce some of the concepts of the Chalk library
# by mimicking the shape of the roman numberal I, V, and X.

# ![](part0.png)

# Chalk uses basic shapes to build up compositional diagrams. For instance here
# is a filled rectangle.

column = rectangle(1, 4).fill_color(black)
column

# Each diagram has style properties and an "envelope" that describes
# its boundaries. Envelopes are a bit complex, they roughly are the
# the bounding box of the diagram.

column.show_envelope()

# The benefit of the envelope representation is that it behaves more
# intuitively under affine transformations like rotation. 

diamond = rectangle(1, 1).fill_color(black).rotate_by(1 / 8)
diamond.show_envelope()


# We can easily combine diagrams. To see what a combination will look
# like we can use the `show_beside` method. You give it a vector along
# which to combine.

column.show_beside(diamond, -unit_y)

# We can also update the envelope of diagrams before combination.
# For instance here we substitute a small envelope for overlap.

column = column.with_envelope(rectangle(1, 2.5))
column.show_beside(diamond, -unit_y)

# When stacking on top we can use the shortcut `/`. The function `center_xy` resets
# the center to the middle of the envelope.

i = (diamond / column / diamond).center_xy()
i.show_envelope()

# We can also put diagrams next to each other using the `|` notation.

ii = i | i
ii

# We can similar create diagrams for the other roman numerals.
# The `align` functions also re-center diagrams.

v = i.align_b() + rectangle(1.5, 1).fill_color(black).align_bl()
v


# Changing the center allows us to use `+` which joins diagrams together at the origin.


v = (v.align_br() + i.align_b()).center_xy()
v


# Creating X is a bit harder. We use two transformations to help us out.
# Using `translate` helps us nudge diagrams away from the origin.


ddiamond = (diamond | diamond).translate(-0.5, 0)
ddiamond.show_origin()

# Using `shear` lets us create a center line with a diagonal slash.

mid = rectangle(2, 0.5).fill_color(black).shear_x(-0.2)
mid


# We can then combine these complex diagrams together.

x = ((ddiamond / column / ddiamond).center_xy() + mid).center_xy()
x


# Compositionality is fun. We can take these shapes and make numbers from 1-12.

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


part0 = concat(
    [
        n.center_xy().scale(0.05).translate_by(-unit_y).rotate_by(-i / 12)
        for i, n in enumerate(numbers)
    ]
)

set_svg_height(500)
part0.show_origin()


# ## Inner Pattern

# ![](part1.png)

# This inner patten is a bit more complex. We are going to need more than just
# simple shapes to draw it. 

# To start, let us make a function for rotational symmetry. 

def rot_cycle(d: Diagram, times: int) -> Diagram:
    "Rotate diagram around a circle."
    return concat(d.rotate_by(i / times) for i in range(times))


# To try it out, we make the inner pattern by making a circle and
# rotating it around 12 times.

set_svg_height(200)

width = -4.4
inner_circle = rot_cycle(circle(1.1).translate(0, width), 12).rotate_by(
    (1 / 12) / 2
) + circle(3).fill_color(black)
inner_circle


# Now we want to trace a trail that looks like the inner patttern.
# There is no magic here, just a little geometry to guess the shapes.
# Vectors `unit_y` and `unit_x` are geometric helpers.

u45 = unit_x.rotate(-45)
u60 = unit_x.rotate(60)
diffy = abs(u45.y / u60.y)
diffx = diffy * abs(u60.x / u45.x)

# A `Trail` is a sequence of vectors drawn in order.
# Once you are done drawing one you can use `stroke` to
# make it a diagram. We start at the top left and draw
# downward. 

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
y.show_origin()


# To draw the curve we use `arc_between` which
# connects two points with a specified radius.

curve = 0.5
under_arc = arc_between(-unit_x, 2 * -unit_y, curve).align_tr()
under_arc.show_origin()

# We then combine them to the right scale.

pattern = (y.scale(3) / under_arc).align_r()
pattern.show_origin()

# And then use reflection to double the pattern.

pattern = (pattern + pattern.reflect_x()).align_b()
pattern


# We can then rotate this to create the whole pattern. We set the
# fudge factor above to make the pattern connect.

set_svg_height(500)
part1 = inner_circle + rot_cycle(pattern.translate(0, width - 1), 12)
part1 = part1.line_color(gold).line_width(0.2)
part1

# Looks pretty close to the original!

# ![](part1.png)


# ## Outer Bands


# ![](part2.png)


# With the functions we have so far it is not so hard to do the rest of the main
# clock-face. The main point worth noting is that we can build each part without
# needing to know the sizes of the others. This makes it easy to debug and update.


# The first band is two circles with a black dots. Nothing new.

set_svg_height(400)

dots = rectangle(0.05, 0.05).fill_color(black)
band1 = (circle(1.1).line_width(0.1) + circle(1)).line_width(0.2) + rot_cycle(
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


# We use this to put the numbers in a circle.

band2 = fit_in(circle(1.4), part0, 0.1) + circle(1)
band2

# We then draw thin lines in this region.

def thin_line(h):
    return rectangle(h, 0.001).fill_color(black).center_xy().line_width(0.01)

lines = thin_line(0.4)
band2 = band2 + rot_cycle(lines.translate(1.2, 0), 48)
band2


# Band 3 has a little jewel cross. We can draw this with shapes.

diamond = rectangle(1, 1).rotate_by(1 / 8)
s = (
    (diamond | rectangle(2, 1).with_envelope(rectangle(1, 1)) | diamond)
    .fill_color(black)
    .center_xy()
)
s = s.scale_y(0.5) + s.rotate_by(0.25).scale_y(0.75).scale_x(0.25)
jewel = s.rotate_by(0.25)
jewel


# Draw the outlines

a = 1.8
c = 1.7
b = 1.6
band3 = (
    circle(2.0).line_width(0.7)
    + (circle(a) + circle(b)).line_width(0.3)
    + circle(1.4).line_width(0.4)
)
band3

# Add the thin lines.

band3 = (
    band3
    + rot_cycle(thin_line(0.4).line_width(0.3).translate(c, 0), 60)
    + rot_cycle(thin_line(0.6).translate(c, 0).rotate_by(1 / 48), 48)
)
band3

# Add the jewel.

band3 = band3 + rot_cycle(
    jewel.center_xy().scale_uniform_to_x((a - b) * 2).translate(0, c), 12
)
band3

# And voila.

part2 = fit_in(band3, fit_in(band2, band1))
set_svg_height(500)
part2

part2.render_pdf("part2.pdf", height=500)

# Looks pretty close to the original!

# ![](part2.png)

# ## Frame

# !<img style="height:500px;width:500px;max-width:auto;" src="bigben.png">


# The whole clock is surrounded by a thick gold frame with some ornamentation. We start with the outer box.

r = rectangle(1, 1).fill_color(black).line_color(gold).line_width(0.6)
r

# Next we do each of the outer corners by themselves. We first make a sloping triangle shape using trails and arc_between.

corner = (
    Trail([unit_x]).stroke().align_l()
    + Trail([-unit_y]).stroke()
    + arc_between((0, -1), (1, 0), -0.2)
).line_width(0.2)
corner


# Internally there is a little golden decoration. To make this we use the `juxtapose` function
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
decal.show_origin()


# We then add a black circle behind it to emphasize details.

decal = decal.line_color(gold).line_width(0.2)
decal = circle(2).fill_color(black) + decal
decal

# To add the other shapes with we `arc` which draws part of a circle.

marc = arc(2 / 2, 90, 3.5 * 90)
decal = concat(
    [
        decal,
        decal.juxtapose(marc.align_t(), unit_x),
        decal.juxtapose(marc.rotate_by(1 / 2).align_r(), -unit_y),
    ]
).line_width(0.2)
decal


# We scale the decoration to fit in the corner we created.

fudge = 0.57
corner = corner.align_bl() + decal.scale_uniform_to_x(fudge).align_bl()
corner = corner.align_tr().translate(-0.4, 0.4).line_color(gold)
corner.show_origin()

# The corners are rotationally symmetric.

corner4 = rot_cycle(corner, 4)
corner4

# Putting it together gives the outer frame.

inner = (
    circle(1)
    .line_width(0.3)
    .line_color(gold)
    .fill_color(black)
    .scale_uniform_to_x(1 - 0.04)
)
part3 = fit_in(r, corner4, 0.05) + inner
part3

# !<img style="height:500px;width:500px;max-width:auto;" src="bigben.png">

# ## Clock Hands

# ![](part1.png)


# To make the clock hands we just trace a path. We `make_path` and give
# it a list of coordinates. We then reflect since it is symmetric. 

hand = make_path(
    [(2, -0.5), (1, -0), (0.4, 20), (0, 21), (0, -1.5), (0.5, -1), (2, -0.5)]
).fill_color(black)
hand = (hand + hand.reflect_x()).translate(0, -4).line_width(0.1)
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
hand2 = (hand2 + hand2.reflect_x()).translate(0, -3).line_width(0.1).line_color(grey)
hand2.show_origin()

# We then overlay them as in the picture at the right scales.

part4 = hand2.scale_uniform_to_y(0.5).rotate_by(0.07) + hand.scale_uniform_to_y(1.0)
part4


# ## All Together

# Our final picture overlays each of the three parts using our fitting functions.
# Each part was done separately, but they all click together to make the final image. 

final = (
    part3
    + fit_in(inner, (fit_in(part2, part1)), 0.0)
    + part4.scale_x(0.8).scale(0.55).rotate_by(0.10)
)

final.render_svg("chalk_bigben.svg", height=500)

# !<img style="height:500px;width:500px;max-width:auto;" src="chalk_bigben.svg">

# !<img style="height:500px;width:500px;max-width:auto;" src="bigben.png">
