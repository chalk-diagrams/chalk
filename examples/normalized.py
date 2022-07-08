from chalk import *


d = triangle(1).line_width(0.1) / triangle(0.5).line_width(0.1).scale(2) / triangle(0.5).line_width(0.1).scale_x(2).scale_y(2) / triangle(1)

height = 100
d.render_svg("examples/output/normalized.svg", height)
d.render("examples/output/normalized.png", height)
d.render_pdf("examples/output/normalized.pdf", height)


