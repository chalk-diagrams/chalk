import pdb
from colour import Color
from chalk import *


def example1():
    dia1 = square(1).translate(0, -4).named(Name("bob")) | circle(1).named(Name("joe"))
    dia1 = dia1.connect(Name("bob"), Name("joe"))
    
    dia2 = square(1).named(Name("bob")).translate(0, -4) | circle(1).named(Name("joe"))
    dia2 = dia2.connect(Name("bob"), Name("joe"))
    
    dia = hcat([dia1, dia2], sep=2)
    dia.render_svg("examples/output/subdiagrams-1.svg")


def example2():
    red = Color("red")

    def attach(subs, dia):
        sub1, sub2 = subs
        p1 = tuple(sub1.get_location())
        p2 = tuple(sub2.get_location())
        return dia + make_path([p1, p2]).line_color(red)


    def squares():
        s = square(1)
        return (
            (s.named(Name("NW")) | s.named(Name("NE"))) /
            (s.named(Name("SW")) | s.named(Name("SE"))))


    dia = hcat([squares().qualify(Name(i)) for i in range(5)], sep=0.5)
    pairs = [
        [Name(0) + Name("NE"), Name(2) + Name("SW")],
        [Name(1) + Name("SE"), Name(4) + Name("NE")],
        [Name(3) + Name("NW"), Name(3) + Name("SE")],
        [Name(0) + Name("SE"), Name(1) + Name("NW")],
    ]

    for pair in pairs:
        dia = dia.with_names(pair, attach)

    dia = dia.show_labels(0.3)
    dia.render_svg("examples/output/subdiagrams-2.svg")


def example3():
    root = circle(1).named(Name("root"))
    leaves = hcat([circle(1).named(Name(c)) for c in "abcde"], sep=0.5).center()

    def connect(subs, nodes):
        subp, subc = subs
        pp = tuple(subp.boundary_from(unit_y))
        pc = tuple(subc.boundary_from(-unit_y))
        return nodes + make_path([pp, pc])

    nodes = root / vstrut(2) / leaves

    for c in "abcde":
        nodes = nodes.with_names([Name("root"), Name(c)], connect)

    nodes.render_svg("examples/output/subdiagrams-3.svg")


example1()
example2()
example3()
