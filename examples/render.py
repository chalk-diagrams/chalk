from chalk import *
import chalk.transform as tx
from colour import Color
import random

from chalk.shapes import set_offset
#import numpy as np
import jax.numpy as np
import jax
from chalk.core import ToList
from chalk.trace import Trace
import jax
#jax.config.update("jax_debug_nans", True)
# blue = Color("#005FDB")
# d =  rectangle(10, 10).rotate(30) #| rectangle(10, 10).align_t()  | circle(50).align_t()

# d = hcat(rectangle(1 + 5*random.random(), 1 + 5*random.random()).rotate_by(random.random()).fill_color(blue)
#          for i in range(1, 20)).line_color(Color("black"))
# d = d.fill_color(Color("orange")).line_color(Color("black")).align_tl().scale(10).translate(10,5)


def render_line(splits, mask, l):
    scene = np.zeros(l)
    split_int = np.floor(splits).astype(int)
    loc = np.arange(splits.shape[0]) % 2
    scene = scene.at[split_int * mask].add(2 * (1 - loc)  - 1)
    scene = scene.at[0].set(0)
    scene = np.cumsum(scene)
    scene = scene.at[split_int * mask].set(1.0 * (1 - loc) + (2 * loc - 1) * ((splits) - split_int))
    scene = scene.at[0].set(0)
    return jax.vmap(lambda s: scene[s + samples] @ kernel(0))(np.arange(scene.shape[0]))

@jax.custom_vjp
def test(x):
    return x

def f_fwd(x):
    return test(x), ()

def f_bwd(res, g):
    print(g)
    return g,

test.defvjp(f_fwd, f_bwd)

@jax.custom_vjp
def split_line(splits, mask, l):
    return l

def f_fwd(splits, mask, l):
    return l, (splits, mask, l)

def f_bwd(res, g):
    splits, mask, l = res
    split_int = np.floor(splits).astype(int) 
    def grad_p(s, s_off):
        off = s_off - s
        v = g[s + samples]
        return (v * (l[s-1] - l[s+1])).sum(-1) @ kernel(off)  
    r = jax.vmap(grad_p, in_axes=(0, 0))(split_int, splits) * mask
    return r, None, g

split_line.defvjp(f_fwd, f_bwd)

kern = 11
samples = np.arange(kern) - (kern//2)
def kernel(offset):
    off_samples = samples - offset
    gaussian_kernel = kern - np.abs(off_samples)
    return np.maximum(0, gaussian_kernel / (kern - np.abs(samples)).sum())

def render_out(d, x=200, y=200):
    # def color_in(ps, m): #, color):
    #     # counter = np.zeros(out.shape[:2])
    #     # def color_row(i, counter):
    #     #     return render_line(ps[i], m[i], counter)
    #     return 

    ls = d.accept(ToList(), tx.ident)
    msaa = np.array([0.5]) # np.array([0,2,1,3]) / 4 + 1 / 8.0
    pt = P2(-1 + tx.np.zeros(y), tx.np.arange(y) + 0.001)
    v = unit_x.repeat(y, axis=0)
    #final = final.at[:, :].set(Color("white").rgb)
    final = np.zeros((x, y, 3))
    final = final.at[:, :].set(Color("white").rgb)
    # ps, m = [], []
    # color = []
    for p in ls:
        trace = p.get_trace()
        ps0, m0 = trace(pt, v)
        out = jax.vmap(render_line, 
                       in_axes=(0, 0, None))(ps0, m0, y)
        
        ps0, m0 = trace(pt[:, [1, 0, 2], :], v[:, [1, 0, 2], :])
        out2 = jax.vmap(render_line, 
                       in_axes=(0, 0, None))(ps0, m0, x).T
        out = out * 0.5 + out2 * 0.5
        out = jax.lax.stop_gradient(out)

        color = p.style.fill_color_
        final = out[..., None] * color + (1-out[..., None]) * final

    for p in ls:
        trace = p.get_trace()
        ps0, m0 = trace(pt, v)
        final = jax.vmap(split_line, in_axes=(0, 0, 0))(ps0, m0, final)
        ps0, m0 = trace(pt[:, [1, 0, 2], :], v[:, [1, 0, 2], :])
        final = jax.vmap(split_line, in_axes=(0, 0, 0))(ps0, m0, final.transpose(1, 0, 2)).transpose(1, 0, 2)

        # out = color_in(ps0, m0)
        #fill = np.zeros((x, y, 3))
        #lines = np.zeros((x, y, 3))
        #lines2 = np.zeros((x, y, 3))

        # ps.append(ps0)
        # m.append(m0)
        # #set_offset(0)
        # color.append(p.style.fill_color_)
    # ps = np.stack(ps, axis=0)
    # m = np.stack(m, axis=0)
    # color = np.stack(color, axis=0)
    # print(ps.shape)
    # final = final + color_in(ps, m, final, color)
    #     #weights = final + color_in(ps0, m0, final) #color)
        # final = weights # final * (1- weights) + fill * weights
        # ps0, m0 = trace(pt[:, [1, 0, 2], :], v[:, [1, 0, 2], :])
        # weights2 = color_in(ps0, m0, final)
        # weights2 = weights2.transpose(1, 0)
        # final = final * 0.5 + weights2 * 0.5
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