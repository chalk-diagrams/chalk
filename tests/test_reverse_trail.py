from hypothesis import given
from hypothesis.strategies import (
    DrawFn,
    composite,
    integers,
    one_of,
    sampled_from,
)

import chalk
from chalk import V2, Trail, unit_x
from chalk.shapes.arc import arc_seg_angle
from chalk.shapes.segment import seg


@composite
def vectors(draw: DrawFn) -> V2:
    x = draw(integers(-2, 2))
    y = draw(integers(-2, 2))
    return V2(x, y)


@composite
def transforms(draw: DrawFn) -> chalk.transform.Affine:
    v2 = draw(vectors())
    return draw(
        sampled_from(
            [
                chalk.transform.Affine.scale(unit_x),
                chalk.transform.Affine.scale(v2),
                chalk.transform.Affine.translation(v2),
                chalk.transform.Affine.rotation(v2.angle),
            ]
        )
    )


@composite
def segment(draw: DrawFn) -> Trail:
    dx = draw(integers())
    dy = draw(integers())
    return seg(V2(dx, dy))


@composite
def arc(draw: DrawFn) -> Trail:
    angle = draw(integers(-360, 360))
    dangle = draw(integers(0, 360))
    return arc_seg_angle(angle, dangle)


small_nat = integers(min_value=1, max_value=10)


@composite
def trails(draw: DrawFn) -> Trail:
    parts = (
        draw(one_of(segment(), arc())).apply_transform(draw(transforms()))
        for _ in range(draw(small_nat))
    )
    return sum(parts, Trail.empty())


@given(trails())
def test_involution(t: Trail) -> None:
    "Reverse should be an involution."
    assert t.reverse().reverse() == t


@given(trails(), trails())
def test_order(t1: Trail, t2: Trail) -> None:
    assert (t1 + t2).reverse() == t2.reverse() + t1.reverse()


# Other possible tests?
# - End point of reversed trail should correspond to start point of original
#   trail (and vicevers);
# - The shape of the trail is preserved.
