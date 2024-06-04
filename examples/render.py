from chalk import *
import chalk.transform as tx
from colour import Color
import random

from chalk.shapes import set_offset
#import numpy as np
import jax.numpy as np
import jax
from chalk.backend.cairo import ToList
from chalk.trace import Trace
import jax
#jax.config.update("jax_debug_nans", True)
# blue = Color("#005FDB")
# d =  rectangle(10, 10).rotate(30) #| rectangle(10, 10).align_t()  | circle(50).align_t()

# d = hcat(rectangle(1 + 5*random.random(), 1 + 5*random.random()).rotate_by(random.random()).fill_color(blue)
#          for i in range(1, 20)).line_color(Color("black"))
# d = d.fill_color(Color("orange")).line_color(Color("black")).align_tl().scale(10).translate(10,5)

@jax.custom_vjp
def make_scene(splits, mask, scene):
    split_int = np.floor(splits).astype(int)
    loc = np.arange(splits.shape[0]) % 2
    scene = scene.at[split_int * mask].add(2 * (1 - loc)  - 1)
    scene = scene.at[0].set(0)
    scene = np.cumsum(scene)
    scene = scene.at[split_int * mask].set(1.0 * (1-loc) + (2 * loc - 1) * ((splits) - split_int))
    scene = scene.at[0].set(0)
    out = jax.vmap(lambda s: scene[s + samples] @ kernel(0))(np.arange(scene.shape[0]))
    return scene, out

def f_fwd(x, mask, scene):
    nscene, out = make_scene(x, mask, scene)
    return (nscene, out), (x, mask, nscene)

def f_bwd(res, g):
    _, g = g
    splits, mask, scene = res
    split_int = np.floor(splits).astype(int) 
    def grad_p(s, s_off):
        off = s_off - s
        v = g[s + samples]
        return (v @ kernel(off)) * (scene[s-1] - scene[s+1])
    r = jax.vmap(grad_p, in_axes=(0, 0))(split_int, splits) * mask
    return r, None, None

make_scene.defvjp(f_fwd, f_bwd)

kern = 11
samples = np.arange(kern) - (kern//2)
def kernel(offset):
    off_samples = samples - offset
    gaussian_kernel = kern - np.abs(off_samples)
    return np.maximum(0, gaussian_kernel / (kern - np.abs(samples)).sum())

def render_out(d, x=200, y=200):
    t = 4 * x
    def color_in(ps, m, out): #, color):
        counter = np.zeros(out.shape[:2])
        def color_row(i, counter):
            idx = i  // (t // y)
            return make_scene(ps[i], m[i], counter)
        i = np.arange(1, ps.shape[0], 4)
        scene, out1 = jax.vmap(color_row, in_axes=(0, 0))(i, counter[i // 4])
        # out1 = jax.vmap(lambda c: 
        #                jax.vmap(lambda s: c[s + samples] @ gaussian_kernel)(np.arange(x)))(counter)
        #out = out1[..., None] * np.array(color)
        return out1 


    ls = d.accept(ToList(), tx.ident)
    msaa = np.array([0,2,1,3]) / 4 + 1/ 8.0
    pt = P2(-1 + tx.np.zeros(t) + msaa.repeat(x), 
            tx.np.arange(t) * x / t + 0.001)
    v = unit_x.repeat(t, axis=0)

    final = np.zeros((x, y))
    #final = final.at[:, :].set(Color("white").rgb)

    for p in ls:
        #fill = np.zeros((x, y, 3))
        #lines = np.zeros((x, y, 3))
        #lines2 = np.zeros((x, y, 3))
        trace = p.get_trace()
        set_offset(0)
        #color = p.style.fill_color_.rgb
        ps0, m0 = trace(pt, v)
        weights = color_in(ps0, m0, final) #color)
        final = weights # final * (1- weights) + fill * weights
        ps0, m0 = trace(pt[:, [1, 0, 2], :], v[:, [1, 0, 2], :])
        weights2 = color_in(ps0, m0, final)
        weights2 = weights2.transpose(1, 0)
        final = final * 0.5 + weights2 * 0.5
    return final


def render(d, name, x=200, y=200):
    t = 4 * x
    def color_in(ps, m, out, color):
        counter = np.zeros(out.shape[:2])
        def color_row(i, counter):
            idx = i  // (t // y)
            return make_scene(ps[i], m[i], counter)
        i = np.arange(1, ps.shape[0], 4)
        counter = jax.vmap(color_row, in_axes=(0, 0))(i, counter[i])
        #out = counter[..., None] * np.array(color)
        out1 = jax.vmap(lambda c: 
                       jax.vmap(lambda s: c[s + samples] @ gaussian_kernel)(np.arange(x)))(counter)
        out = out1[..., None] * np.array(color)
        return out # counter[..., None]

        # for i in range(2, ps.shape[0], 4):
        #     #last = pt[i][None]
        #     #count = 0
            
        #     #counter = counter.at[idx].set(0)

        #     counter = counter.at[idx].set(make_scene(ps[i], m[i], counter[idx]))
        #     # for j in range(ps.shape[1]):
            #     if m[i, j]:
            #         if count % 2 == 1:
            #             counter[i  // (t // y), int(ps[i, j-1]): max(int(ps[i, j]), int(ps[i, j-1])+1)] += 1

            #         count += 1
                    #next = tx.translation(V2(ps[i, j], 0)) @ pt[i][None]
                    # scan = Path.from_points([last, next+1e-2]).stroke()
                    # scan = scan.line_color(Color("red") if count % 2 == 0 else Color("blue"))

                    # d2 += scan
                    # last = next
        # d2.render_svg("examples/output/trace_circle.svg")
        # exit()
        #print(counter.shape, color.shape)


    ls = d.accept(ToList(), tx.ident)
    msaa = np.array([0,2,1,3]) / 4 + 1/ 8.0
    pt = P2(-1 + tx.np.zeros(t) + msaa.repeat(x), 
            tx.np.arange(t) * x / t + 0.001)
    
    v = unit_x.repeat(t, axis=0)

    final = np.zeros((x, y, 3))
    final = final.at[:, :].set(Color("white").rgb)

    for p in ls:
        
        fill = np.zeros((x, y, 3))
        lines = np.zeros((x, y, 3))
        lines2 = np.zeros((x, y, 3))
        trace = p.get_trace()

        set_offset(0)
        color = p.style.fill_color_.rgb
        ps0, m0 = trace(pt, v)
        weights = color_in(ps0, m0, fill, color)
        final = final * (1- weights) + fill * weights

        if False: # stroke
            color = p.style.line_color_
            if color is None:
                color = Color("black")
            color = color.rgb
            for p2 in p.shape.split():

                trace = p2.apply_transform(p.transform).get_trace()
                set_offset(-0.5)
                a = trace(pt, v)
                set_offset(0.5)
                b = trace(pt, v)
                ps, m = Trace.combine(a, b)
                weights = color_in(ps, m, lines, color)
                final = final * (1- weights) + weights * lines

                set_offset(-0.5)
                a = trace(pt[:, [1, 0, 2], :], v[:, [1, 0, 2], :])
                set_offset(0.5)
                b = trace(pt[:, [1, 0, 2], :], v[:, [1, 0, 2], :])
                ps, m = Trace.combine(a, b)
                weights2 = color_in(ps, m, lines2.transpose(1, 0, 2), color)
                weights2 = weights2.transpose(1, 0, 2)
                final = final * (1- weights2) + weights2 * lines2

def render(d, name, x, y):
    show(render_out(d, x, y), name)

def show(final, name):
    import matplotlib.pyplot as plt
    im = plt.imshow(final)
    plt.savefig(name)


# ps, m = trace.trace_p(pt, v)
# print(ps, m)
# for i in range(ps.shape[0]):
#     if m[i]:
#         d += Path.from_points([pt[i:i+1], ps[i:i+1]]).stroke()

#d.render_svg("examples/output/trace_circle.svg")
# ps, m = trace.trace_p(P2(0, 0).repeat(30, axis=0), tx.polar(tx.np.arange(0, 30)) * 1.)

# for i in range(ps.shape[0]):
#     if m[i]:
#         d += seg(ps[i]).stroke()

# ps, m = trace.max_trace_p(P2(0, 0).repeat(60, axis=0), tx.polar(tx.np.arange(-30, 30) + 5) * 1.)
# for i in range(ps.shape[0]):
#     if m[i]:
#         d += seg(ps[i]).stroke().line_color(Color("red"))

# d.render_svg("examples/output/trace_circle.svg")