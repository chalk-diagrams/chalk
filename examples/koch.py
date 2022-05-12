from chalk import *
from chalk.transform import *


def koch(n):
    if n == 0:
        return Trail.from_path(hrule(5))
    else:
        return (
            koch(n - 1).scale(1 / 3)
            + koch(n - 1).scale(1 / 3).rotate_by(-1 / 6)
            + koch(n - 1).scale(1 / 3).rotate_by(+1 / 6)
            + koch(n - 1).scale(1 / 3)
        )


d = vcat(koch(i).stroke().line_width(0.01) for i in range(1, 5))

height = 512
d.render_svg("examples/output/koch.svg", height)
d.render("examples/output/koch.png", height)
