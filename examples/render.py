from chalk import *
import chalk.transform as tx
from colour import Color
import random

from chalk.shapes import set_offset
import numpy as np

from chalk.backend.cairo import ToList


# blue = Color("#005FDB")
# d =  rectangle(10, 10).rotate(30) #| rectangle(10, 10).align_t()  | circle(50).align_t()

# d = hcat(rectangle(1 + 5*random.random(), 1 + 5*random.random()).rotate_by(random.random()).fill_color(blue)
#          for i in range(1, 20)).line_color(Color("black"))
# d = d.fill_color(Color("orange")).line_color(Color("black")).align_tl().scale(10).translate(10,5)





def combine(p1, p2):
    ps, m = p1
    ps2, m2 = p2
    ps = np.concatenate([ps, ps2], axis=1)
    m = np.concatenate([m, m2], axis=1)
    ad = tx.np.argsort(ps + (1-m) * 1e10, axis=1)
    ps = tx.np.take_along_axis(ps, ad, axis=1) 
    m = tx.np.take_along_axis(m, ad, axis=1) 
    return ps, m

def render(d, name, x=200, y=200):
    t = 4 * y

    def color_in(ps, m, out, color):
        counter = np.zeros(out.shape[:2])
        for i in range(ps.shape[0]):
            #last = pt[i][None]
            count = 0 
            for j in range(ps.shape[1]):
                if m[i, j]:
                    if count % 2 == 1:
                        counter[i // (t // y), round(ps[i, j-1]): round(ps[i, j])] += 1
                    # next = tx.translation(V2(ps[i, j], 0)) @ pt[i][None]
                    # scan = Path.from_points([last, next+1e-2]).stroke()
                    # scan = scan.line_color(Color("red") if count % 2 == 0 else Color("blue"))
                    count += 1
                    # d += scan
                    # last = next
        out[:, :, :3] = np.where((counter > 0) [..., None], color, 0)
        return (counter / 4)[..., None]


    ls = d.accept(ToList(), tx.ident)
    pt = P2(-1 + tx.np.zeros(t) + 0.25 + 0.5 * tx.np.arange(t) % 2, 
            tx.np.arange(t) * x / t + 0.001)
    v = unit_x.repeat(t, axis=0)

    final = np.zeros((x, y, 3))
    final[:, :] = Color("white").rgb

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
            ps, m = combine(a, b)
            weights = color_in(ps, m, lines, color)
            final = final * (1- weights) + weights * lines

            set_offset(-0.5)
            a = trace(pt[:, [1, 0, 2], :], v[:, [1, 0, 2], :])
            set_offset(0.5)
            b = trace(pt[:, [1, 0, 2], :], v[:, [1, 0, 2], :])
            ps, m = combine(a, b)
            weights2 = color_in(ps, m, lines2.transpose(1, 0, 2), color)
            weights2 = weights2.transpose(1, 0, 2)
            final = final * (1- weights2) + weights2 * lines2



    import matplotlib.pyplot as plt
    im = plt.imshow(final + 0.,  cmap='Greys',  interpolation='nearest')
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