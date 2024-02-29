from dataclasses import dataclass
from PIL import Image as PILImage
from chalk import *
from colour import Color
import numpy as np
from numpy.typing import ArrayLike

h = hstrut(2.5)
papaya = Color("#ff9700")
white = Color("white")
black = Color("black")


def lookAt(eye: ArrayLike, center: ArrayLike, up: ArrayLike):
    "Python version of the haskell lookAt function in linear.projections"
    f = (center - eye) / np.linalg.norm(center - eye)
    s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
    u = np.cross(s, f)
    return np.array([[*s, 0], [*u, 0], [*-f, 0], [0, 0, 0, 1]])


def scale3(x, y, z):
    return np.array([[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]])


@dataclass
class D3:
    x: float
    y: float
    z: float

    def to_np(self):
        return np.array([self.x, self.y, self.z])


V3 = D3


def homogenous(trails: List[List[D3]]):
    "Convert list of directions to a np.array of homogenous coordinates"
    return np.array([[[*o.to_np(), 1] for o in offsets] for offsets in trails])


def cube():
    "3 faces of a cube drawn as offsets from the origin."
    return homogenous(
        [
            [D3(*v) for v in offset]
            for offset in [
                [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)],
                [(1, 0, 0), (0, 0, 1), (-1, 0, 0), (0, 0, -1)],
                [(0, 0, 1), (0, 1, 0), (0, 0, -1), (0, -1, 0)],
            ]
        ]
    )


def to_trail(trail: ArrayLike, locations: ArrayLike):
    return [
        (
            Path(
                [Trail.from_offsets([V2(*v[:2]) for v in trail]).close().at(V2(*l[:2]))]
            ),
            l[2],
        )
        for l in locations
    ]


def project(projection, shape3, positions):
    p = homogenous([positions for _ in range(shape3.shape[0])])
    locations = p @ projection.T
    trails = shape3 @ projection.T
    return [out for t, l in zip(trails, locations) for out in to_trail(t, l)]


# Create Data
x = np.random.rand(20, 30, 40) > 0.9
a, b, c = x.nonzero()

# Big Cube
s = scale3(*x.shape)
big_cube = cube() @ s.T
s_ = x.shape

# Isometric projection of tensor
projection = lookAt(
    V3(s_[1] + s_[2], s_[0] + s_[2], s_[0] + s_[1]).to_np(),
    V3(0, 0, 0).to_np(),
    V3(0, 0, 1).to_np(),
)
outer = project(projection, big_cube, [V3(0, 0, 0)])
d = (
    concat([p.stroke().fill_color(white).fill_opacity(0.1) for p, _ in outer])
    .line_width(0.2)
    .line_color(white)
)

cubes = project(projection, cube(), [V3(x, y, z) for x, y, z in zip(a, b, c)])
cubes.sort(key=lambda x: x[1], reverse=True)
d2 = concat([p.stroke() for p, _ in cubes])
d = d2.fill_color(papaya).fill_opacity(0.9).line_width(0.05).with_envelope(d) + d
d.render("output/tensor2.png", 500)
d.render_svg("output/tensor2.svg", 500)
