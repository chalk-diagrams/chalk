import jax
import jax.numpy as np
import os
os.environ["CHALK_JAX"] = "1"
# from jaxtyping import install_import_hook
# with install_import_hook("chalk", "typeguard.typechecked"):
#     import chalk 
from chalk import *
from colour import Color
import numpy as onp
import optax

@jax.vmap
def inner(i):
    # return (circle(0.3).fill_color(np.ones(3) * (6 - i) / 6))
    return (circle(0.3 * i / 6).fill_color(np.ones(3) * i / 6) + 
            square(0.1).fill_color("white"))

inside = hcat(inner(np.arange(2, 6)))
inside.get_trace()(P2(0, 0), V2(1, 0))
inside.render("/tmp/t.png")



@jax.vmap
def outer(j):
    @jax.vmap
    def inner(i):
        # return (circle(0.3 * i / 6).fill_color(np.ones(3) * i / 6))
        return (circle(0.3 * i / 6).fill_color(np.ones(3) * i / 6) + 
                square(0.1).fill_color("white"))
    inside = inner(np.arange(2, 6))
    return inside
out = outer(np.arange(1, 5))
print("My Size", hcat(out).size())
d = vcat(hcat(out))
print("PRIMS:", [prim.order for prim in  d.accept(chalk.core.ToListOrder(), tx.X.ident).ls])

#arc_seg(V2(0, 1), 1).stroke().render("/tmp/t.png", 64)

#jax.tree.map(lambda x: print(x.shape), d)
d.render("/tmp/t.png")
exit()
# print(out.get_trace()(P2(0, 0), V2(1, 0)))
# d = (rectangle(10, 2).fill_color("white") + out)
# d.render("temp.png", 64)


seed = 1701
size = 50
connects = 1
around = 5
color = np.stack([chalk.style.to_color(c) for c in  Color("red").range_to("blue", size // around)],
                 axis=0)

key = jax.random.PRNGKey(0)
matrix = jax.random.uniform(key, (size, 2)) * 4 - 2
#all = jax.random.categorical(key, np.ones((size, connects, size)), axis=-1)
all = np.stack([np.minimum(size, np.ones((size,), int) + around * (np.arange(size) // around))  for i in range(connects)], axis=1)

@jax.jit
def graph(x):
    #x = np.minimum(np.maximum(x, 0), 1)
    center =  1/10 * np.abs(x).sum()
    repulse = (1/ size) * ((1 / (1e-3 + np.pow(x[:, None, :] - x, 2).sum(-1))) * (1 -np.eye(size))).sum()

    def dots(p, i):
        d = circle(0.1).translate(p[0], p[1]).fill_color(color[i // around] * np.maximum((i % around) / (around - 1), 0.5))
        return d

    def connect(p):
        eps = 1e-4
        d = make_path([(p[0, 0], p[0, 1]), 
                        (p[1, 0]+ eps, p[1, 1] + eps)])
        return d

    out = jax.vmap(dots)(x, np.arange(size)).with_envelope(empty()).line_width(1)    
    a, b = x[:, None, :].repeat(connects, axis=1), x[all]
    v = np.stack([a, b], axis=-2).reshape((-1, 2, 2))
    spring = size * np.pow(np.pow(v[:, 1] - v[:, 0], 2).sum(-1) - 0.04, 2).sum()
    lines = jax.vmap(connect)(v).with_envelope(empty())
    
    out = lines.line_width(1) + out
    out = rectangle(4, 4).fill_color("white")  + out
    out, h, w = chalk.core.layout_primitives(out, 500)
    score = spring + center + repulse
    return score, (out, h, w)

def opt(x, fn):
    res = []
    fn = jax.jit(jax.value_and_grad(fn, has_aux=True))
    solver = optax.adam(learning_rate=0.3)
    opt_state = solver.init(x)
    for j in range(500):
        print(j)
        value, grad = fn(x)
        score, out = value
        updates, opt_state = solver.update(grad, opt_state, x)
        x = optax.apply_updates(x, updates)

        if True:
            out, h, w = out
            out = jax.tree.map(onp.asarray, out)
            import chalk.transform
            chalk.transform.set_jax_mode(False)
            print("RENDER")
            chalk.backend.svg.prims_to_file(out, f"test.{j:03d}.svg", h, w)
            #chalk.backend.cairo.prims_to_file(out, f"test.{j:03d}.png", h, w)
            chalk.transform.set_jax_mode(True)
        print(score)
opt(matrix, graph)

# score, out = graph(matrix)
# out = out.unflatten(no_map=True)[0]
# print(score)
# print("concat")

# print("LINES")
# #out.render("test.png", 200)

# import chalk.backend.cairo
# out, h, w = chalk.core.layout_primitives(out, 200)
# print("OUT")
# out = jax.tree.map(np.asarray, out)
# chalk.backend.cairo.prims_to_file(out, "test.png", h, w)
