from jaxtyping import install_import_hook
with install_import_hook("chalk", "typeguard.typechecked"):
    import chalk          # Any module imported inside this `with` block, whose

from colour import Color
from chalk import *
import render
import jax
import chalk
import jax.numpy as np
# define some colors
papaya = Color("#ff9700")
blue = Color("#005FDB")



path = "examples/output/intro-01.png"

def animate(fn, t, path, **kwargs):
    import imageio
    frames = jax.vmap(lambda t: chalk.core.flatten(fn(t)))(np.arange(t))
    frames = chalk.core.unflatten(frames)
    path_frame = "/tmp/{:d}-out.png"
    with imageio.get_writer(path, fps=1, **kwargs) as writer:
        z = concat(frames)
        for i, frame in enumerate(frames):
            path = path_frame.format(i)
            frame.with_envelope(z).render(path)
            image = imageio.imread(path)
            writer.append_data(image)

import os
def animate_svg(fn, t, path, **kwargs):
    import imageio
    frames = jax.vmap(lambda t: chalk.core.flatten(fn(t)))(np.arange(t))
    frames = chalk.core.unflatten(frames)
    path_frame = "/tmp/{:d}-out.svg"

    z = concat(frames)
    for i, frame in enumerate(frames):
        p = path_frame.format(i)
        frame.with_envelope(z).render_svg(p)

    os.system(f"svgasm /tmp/*-out.svg -o {path}")

#animate(x, 6, "out.gif")
animate_svg(x, 6, "out.svg")
exit()
hcat(chalk.core.unflatten(out)).render(path, height=64)


exit()
# chalk.core.unf(x(0)).render(path, height=64)

#chalk.core.unflatten(out)[0].render(path, height=64)

path = "examples/output/intro-01.png"
out = jax.vmap(x)(np.array([1,2,3]))


concat(chalk.core.unflatten(out)).render(path, height=64)

for i, d in enumerate(chalk.core.unflatten(out)):
    path = f"examples/output/intro-01-{i}.png"
    d.render(path, height=64)
exit()



d = rectangle(3, 10).fill_color(papaya)
d = d.translate_by(V2(np.array([10, 11]), np.array([11, 12])))
#t = d.get_trace()
print(d)


#d.render(path, height=64)
#render.render(d, "render.png", 200, 200)

#d = regular_polygon(8, 1.5).rotate_by(1 / 16)
t = d.get_trace()
print("trace", t(P2(0, 0.), V2(1., 0.)))
#print("trace", t(P2(1, 1.), V2(0., 1.)))
d.render(path, height=64)
print("first")

# # Alternative, render as svg
path = "examples/output/intro-01.svg"
d.render_svg(path, height=64)

# Alternative, render as pdf
# path = "examples/output/intro-01.pdf"
# d.render_pdf(path, height=64)


path = "examples/output/intro-02.png"
# d = circle(0.5).fill_color(papaya) | square(1).fill_color(blue)
d = circle(0.5).fill_color(papaya) | square(1).fill_color(blue)
d.render(path, height=64)

path = "examples/output/intro-02.svg"
d.show_envelope().render_svg(path, height=64)
# path = "examples/output/intro-02.pdf"
# d.render_pdf(path)

path = "examples/output/intro-03.png"
d = hcat(circle(0.1 * i) for i in range(1, 6)).fill_color(blue)
d.render(path, height=64)

# Alternative, render as svg
path = "examples/output/intro-03.svg"
d.render_svg(path, height=64)

# # Alternative, render as pdf
# path = "examples/output/intro-03.pdf"
# d.render_pdf(path)

path = "examples/output/intro-04.png"
print("sierpinsky")
def sierpinski(n: int, size: int) -> Diagram:
    if n <= 1:
        return triangle(size)
    else:
        smaller = sierpinski(n - 1, size / 2)
        return smaller.above((smaller | smaller).center_xy())

d = sierpinski(5, 4).fill_color(papaya)
# d.render(path, height=256)

# path = "examples/output/intro-04.svg"
# d.render_svg(path, height=256)
import render
print("render")
render.render(d.align_tl().scale_uniform_to_x(200), "render.png", 200, 200)

# path = "examples/output/intro-04.pdf"
# d.render_pdf(path, height=256)
