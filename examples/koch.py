# Based on the following example from the Doodle library:
# https://github.com/creativescala/doodle/blob/master/image/shared/src/main/scala/doodle/image/examples/Koch.scala

import math

from typing import List

import streamlit as st

from diagrams.core import Primitive
from diagrams.point import Point, Vector
from diagrams.shape import Path, PathElement, MoveTo, LineTo
from diagrams import Diagram, rectangle, concat, hcat, vcat, empty


deg_60 = math.pi / 3


def elements(
    depth: int, start: Point, angle: float, length: float
) -> List[PathElement]:
    if depth == 0:
        return [LineTo(start + Vector.from_polar(length, angle))]
    else:
        l_angle = angle - deg_60
        r_angle = angle + deg_60

        third = length / 3
        edge = Vector.from_polar(third, angle)

        mid1 = start + edge
        mid2 = mid1 + edge.rotate(-deg_60)
        mid3 = mid2 + edge.rotate(+deg_60)

        return (
            elements(depth - 1, start, angle, third)
            + elements(depth - 1, mid1, l_angle, third)
            + elements(depth - 1, mid2, r_angle, third)
            + elements(depth - 1, mid3, angle, third)
        )


def koch(depth: int, length: float) -> Diagram:
    origin = Point(0, length / 6)
    shape = Path([MoveTo(origin)] + elements(depth, origin, 0, length))
    return Primitive.from_shape(shape)


# Unfortunately, this implementation doesn't yield the desired result since the
# bounding boxes are not tight.
# def koch2(depth: int):
#     if depth == 0:
#         return hrule(1)
#     else:
#         return hcat(
#             koch2(depth - 1).scale(1 / 3),
#             koch2(depth - 1).scale(1 / 3).rotate_by(-1 / 6).align_b(),
#             koch2(depth - 1).scale(1 / 3).rotate_by(+1 / 6).align_b(),
#             koch2(depth - 1).scale(1 / 3),
#         )


path = "examples/output/koch.png"
diagram = vcat(koch(i, 4) for i in range(1, 5))
diagram.render(path, height=512)
st.image(path)
