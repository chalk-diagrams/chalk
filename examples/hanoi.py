# Based on the following example from Diagrams
# https://archives.haskell.org/projects.haskell.org/diagrams/gallery/Hanoi.html

from PIL import Image as PILImage
from typing import Dict, List, Tuple

from colour import Color  # type: ignore
from chalk import Diagram, rectangle, concat, hcat, vcat


Disk = int
Stack = List[Disk]  # disks on one peg
Hanoi = List[Stack]  # disks on all three pegs
Move = Tuple[int, int]

colors: List[Color] = [
    Color("#9FB4CC"),
    Color("#CCCC9F"),
    Color("#DB4105"),
]
black = Color("black")


def draw_disk(n: Disk) -> Diagram:
    return (
        rectangle(n + 2, 1)
        .fill_color(colors[n])
        .line_color(colors[n])
        .line_width(0.05)
    )
draw_disk(0)


def draw_stack(s: Stack) -> Diagram:
    disks = vcat(map(draw_disk, s))
    post = rectangle(0.8, 6).fill_color(black)
    return post.align_b() + disks.align_b()
draw_stack([0, 1])


def draw_hanoi(state: Hanoi) -> Diagram:
    hsep = 7
    return concat(
        draw_stack(stack).translate(7 * i, 0) for i, stack in enumerate(state)
    )
draw_hanoi([[0], [1, 2], []])

def solve_hanoi(num_disks: int) -> List[Move]:
    def solve_hanoi_1(num_disks, *, source, spare, target):
        if num_disks <= 0:
            return []
        else:
            return (
                solve_hanoi_1(num_disks - 1, source=source, spare=target, target=spare)
                + [(source, target)]
                + solve_hanoi_1(num_disks - 1, source=spare, spare=source, target=target)
            )

    return solve_hanoi_1(num_disks, source=0, spare=1, target=2)


def do_move(move: Move, state: Hanoi) -> Hanoi:
    def remove_disk(src, state):
        disk, *src_new = state[src]
        state_new = state[:src] + [src_new] + state[src + 1 :]
        return disk, state_new

    def add_disk(tgt, disk, state):
        tgt_new = [disk] + state[tgt]
        state_new = state[:tgt] + [tgt_new] + state[tgt + 1 :]
        return state_new

    src, tgt = move
    disk, state1 = remove_disk(src, state)
    state2 = add_disk(tgt, disk, state1)
    return state2


def state_sequence(num_disks: int) -> List[Hanoi]:
    state: Hanoi = [list(range(num_disks)), [], []]
    states: List[Hanoi] = [state]
    for move in solve_hanoi(num_disks):
        state = do_move(move, state)
        states.append(state)
    return states


def draw_state_sequence(seq: List[Hanoi]) -> Diagram:
    return concat(draw_hanoi(state).translate(0, 7.5 * i) for i, state in enumerate(seq))


diagram = draw_state_sequence(state_sequence(3))


path = "examples/output/hanoi.svg"
diagram.render_svg(path, height=700)
path = "examples/output/hanoi.png"
diagram.render(path, height=700)
PILImage.open(path)
