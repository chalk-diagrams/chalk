from chalk import *
from colour import Color

# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Big_Ben_Clock_Face.jpg/612px-Big_Ben_Clock_Face.jpg?20140924183502)

gold = Color("#E7D49C")
white = Color("white")
black = Color("black")

# ## Preliminary


# We start by drawing some basic diagrams. Here is a rectangular column.
# Each diagram has style properties and an "envelope" that describes its
# boundaries. Envelopes are a bit complex, they roughly describe the boundaries
# of the diagram.

column = rectangle(1, 4).fill_color(black)
column.show_envelope(rate=12)


# The benefit of the envelope representation (over something simpler like
# a bounding box) is that it behaves more intuitively under affine transformations
# like rotation. 

diamond = rectangle(1, 1).fill_color(black).rotate_by(1/8)
diamond.show_envelope()


# We can also change the envelope of diagrams to help with object placement.
# For instance here we substitute a small envelope. 

column = column.with_envelope(rectangle(1, 2.5))
column.show_envelope(rate=12)

# Envelopes are used for combining diagrams. We can easily stack our shapes
# using the `above` combinator written with `/`.

i = (diamond / column / diamond).center_xy()
i.show_envelope()

# We can also put diagrams next to each other using the `|` combinator. 

ii = i | i
ii

# We can similar create diagrams for the other roman numerals. 
# The `align` function changes the center (red dot) of a diagram. The `+`
# combinator attaches diagrams at the origin.

v = i.align_b() + rectangle(1.5, 1).fill_color(black).align_bl()
v.show_envelope()

v = (v.align_br() + i.align_b()).center_xy()
v.show_envelope()


# Creating X is a bit harder. We use two transformations to help us out.
# Using `translate` helps us nudge the center alignment of the diamonds over.
# Using `shear` lets us create a center line with a diagonal slash.

ddiamond = (diamond | diamond).translate(-0.5, 0)
ddiamond.show_origin()

x = ddiamond / column /  ddiamond
x = (x.center_xy() + rectangle(2, 0.5).fill_color(black).shear_y(-0.2)).center_xy()
x


# Now we have all of our roman numerals, we can list them in clock-wise order. 

numbers = [x | i | i, i, i | i, i | i | i, i | v, v, v | i, v | i | i, v | i | i | i , i | x, x, x | i]


# We can draw the main clock-face by moving each number to the edge, and then rotating it
# to its location. The `concat` combinator glues each of these together at the origin. 

part0 = concat([n.center_xy().scale(0.05).translate_by(-unit_y).rotate_by(-i / 12) for i, n in enumerate(numbers)])

part0



# Inner design

y = ((v2_stroke(1.5 * unit_x.rotate(-45)).align_br() ) / v2_stroke(0.5 * unit_y)).align_t()
y2 = (v2_stroke(1.5 * unit_x.rotate(-60)).align_br() ).reflect_y().align_t()
y = (y + y2)

env = y.get_envelope()
bottom = env.envelope_v(unit_y)
right = env.envelope_v(unit_x.rotate(-2*60))
marc = arc_between((bottom.x, bottom.y), (right.x, right.y), -0.1)

# y = y + arc
# y
y

marc = arc_between(-unit_x/3, 2* -unit_y/3, 0.5/3)

fig = marc.align_bl()
fig = (y / fig.align_tr()).align_r()
fig = (fig + fig.reflect_x()).align_b().scale(3)


def rot_cycle(d: Diagram, times: int) -> Diagram:
    x = empty()
    for i in range(times):
        x += d.rotate_by(i / times)
    return x


cs = rot_cycle(circle(1.1).translate(0, -4.4), 12).rotate_by((1/12) /2 )
cs = cs + rot_cycle((fig + fig.reflect_x()).align_b().translate(0, -5.4), 12) 
cs = cs + circle(3).fill_color(black)
part1 = cs.line_color(gold)

part1


# Helper



def fit_in(b: Diagram, s:Diagram) -> Diagram:
    m = min([x for x in b.get_trace()(origin, unit_x) if x > 0])
    return b + s.scale_uniform_to_x(2*m)




# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Big_Ben_Clock_Face.jpg/612px-Big_Ben_Clock_Face.jpg?20140924183502)

# Outer design

dots = rectangle(0.05, 0.05).fill_color(black)
band1 =  (circle(1.1) +  circle(1)).line_width(0.03) + rot_cycle(dots.translate(1.05, 0), 12)
band1 

lines = rectangle(0.4, 0.0005).line_width(0.001).fill_color(black)





band2 = circle(1.5) + circle(1.1) + rot_cycle(lines.translate(1.3, 0), 48)  + hours
band2


lines2 = rectangle(0.3, 0.0005).fill_color(black).line_width(0.03)

s = (rectangle(1, 1).rotate_by(1/8) | rectangle(2, 1).with_envelope(rectangle(1, 1)) | rectangle(1, 1).rotate_by(1/8)).fill_color(black).center_xy()
s = s.scale_y(0.5) + s.rotate_by(0.25).scale_y(0.75).scale_x(0.25)
s

jewel = s.rotate_by(0.25).scale_uniform_to_x(0.4)

lines = rectangle(0.6, 0.0005).line_width(0.001).fill_color(black).align_l()

band3 = circle(2.1).line_width(0.1) + (circle(1.9) + circle(1.7)).line_width(0.05) + circle(1.5).line_width(0.08) + rot_cycle(lines.translate(1.5, 0), 60) + rot_cycle(lines2.translate(1.8, 0).rotate_by(1/48), 48) + rot_cycle(jewel.translate(0, 1.8), 12)
band3 


part2 = band3 + band2 + band1
part2


#





# # Outer frame

r = rectangle(1, 1).fill_color(black).line_color(gold).line_width(0.1)
r


decal = rectangle(2, 2).with_envelope(rectangle(0, 0)).fill_color(gold)
c = circle(1).fill_color(black)
decal = concat([decal.juxtapose(c, unit_y),
                decal.juxtapose(c, -unit_y),
                decal.juxtapose(c, unit_x),
                decal.juxtapose(c, -unit_x),
                decal])
outer = decal.get_envelope().width / 2
decal = circle(outer).fill_color(black) + decal
marc = arc(outer/2, 90, 3.5 * 90)
decal = decal + decal.juxtapose(marc.align_t(), unit_x) + decal.juxtapose(marc.rotate_by(1/2).align_r(), -unit_y)
decal = decal.line_width(0.03)
decal


corner = (v2_stroke(unit_x).align_l() + v2_stroke(-unit_y) + arc_between((0, -1), (1, 0), -0.2)).line_width(0.03)
corner = corner.align_bl() + decal.scale_uniform_to_x(0.57).align_bl()
corner = corner.align_tr().translate(-0.4, 0.4).line_color(gold)
corner 


inner = circle(0.485).line_width(0.08).line_color(gold).fill_color(black)
part3 = r + rot_cycle(corner, 4).scale_uniform_to_x(0.96) + inner


part3



# # Hands 

triangle(1).rotate_by(1/12)


x = arc_between((0, -1), (2, 0), -0.2) + arc_between((2, 0), (1, 2), -0.3)  + arc_between((0.4, 20), (0, 20 + 1), 0.1) + make_path([(1, -0.5), (0.4, 20), (0, 21), (0, -0.5), (1, -0.5)]).fill_color(black)
x = x + arc_between((0, -1), (0, 21), 0).line_width(0.1)
x = (x + x.reflect_x()).translate(0, -4)

part4 = x.line_width(0.05)
part4



# ## All Together

fit_in(part2, part1.line_width(0.03))


part3.center_xy().get_trace()(origin, unit_x)

part3 + fit_in(inner, (fit_in(part2, part1.line_width(0.03)))) + part4.scale(0.485 /  part4.get_envelope()(unit_y)  ).rotate_by(0.05)

         

# fit_in(part3,

#        ((fit_in(part2, part1.line_width(0.03))) + part4.scale_uniform_to_y(2.8).rotate_by(0.05)))





# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Big_Ben_Clock_Face.jpg/612px-Big_Ben_Clock_Face.jpg?20140924183502)
