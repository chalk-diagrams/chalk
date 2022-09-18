import math

import pytest
from hypothesis import given
from hypothesis.strategies import (
    DrawFn,
    composite,
    integers,
    lists,
    one_of,
    sampled_from,
)

import chalk
from chalk import (
    P2,
    V2,
    Diagram,
    Trail,
    circle,
    empty,
    make_path,
    origin,
    rectangle,
    unit_x,
    unit_y,
)


@composite
def vectors(draw: DrawFn) -> V2:
    x = draw(integers(min_value=-2, max_value=2).filter(lambda x: x != 0))
    y = draw(integers(min_value=-2, max_value=2).filter(lambda x: x != 0))
    return V2(x, y)


small_nat = integers(min_value=1, max_value=10)


@composite
def trails(draw: DrawFn) -> Trail:
    vs = draw(lists(vectors(), min_size=1))
    return Trail.from_offsets(vs)


@composite
def paths(draw: DrawFn) -> Diagram:
    return draw(trails()).stroke().center_xy()


@composite
def circles(draw: DrawFn) -> Diagram:
    return circle(draw(small_nat))


@composite
def rects(draw: DrawFn) -> Diagram:
    return rectangle(draw(small_nat), draw(small_nat))


@composite
def shapes(draw: DrawFn) -> Diagram:
    return draw(one_of(paths(), rects(), circles()))


@composite
def diagrams(draw: DrawFn) -> Diagram:
    shape = empty()
    for j in range(3):
        lshape = draw(shapes())
        shape += chalk.transform.apply_affine(draw(transforms()), lshape)
    return shape


@composite
def transforms(draw: DrawFn) -> chalk.transform.Affine:
    v2 = draw(vectors())
    return draw(
        sampled_from(
            [
                chalk.transform.Affine.scale(v2),
                chalk.transform.Affine.translation(v2),
                chalk.transform.Affine.rotation(v2.angle),
            ]
        )
    )


@given(diagrams(), vectors())
def test_envelope_trail(diagram: Diagram, vec: V2) -> None:
    "Property -> Envelope bounds trace."
    trace = diagram.get_trace()
    env = diagram.get_envelope()
    ts = trace(P2(0, 0), vec)
    e = env(vec)
    for t in ts:
        assert e == pytest.approx(t) or e > t


@given(diagrams(), vectors())
def test_pad(diagram: Diagram, vec: V2) -> None:
    orig = diagram.get_envelope()(vec)
    p = diagram.pad(2)
    assert p.get_envelope()(vec) == pytest.approx(2 * orig)
    vec = vec.normalized()
    orig = diagram.get_envelope()(vec)
    f = diagram.frame(2)
    assert f.get_envelope()(vec) == pytest.approx(2 + orig)


# Some specific tests.
def test_square() -> None:
    square = make_path([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    env = square.get_envelope()
    assert env(unit_x) == 1
    assert env(2 * unit_x) == 0.5
    assert env(unit_y) == 1
    assert env((unit_x + unit_y).normalized()) == pytest.approx(
        math.sqrt(1 + 1)
    )


def test_circle() -> None:
    d = circle(1)
    env = d.get_envelope()
    assert env(unit_x) == 1
    assert env(2 * unit_x) == 0.5
    assert env(unit_y) == 1
    assert env((unit_x + unit_y).normalized()) == pytest.approx(1)


def test_circle_trace() -> None:
    d = circle(1)
    trace = d.get_trace()
    assert set(trace(origin, unit_x)) == set([-1.0, 1.0])
    assert set(trace(origin, (2 * unit_x))) == set([-0.5, 0.5])
    assert set(trace(origin, unit_y)) == set([-1.0, 1.0])
    trace(origin, (unit_x + unit_y))


def test_path_trace() -> None:
    d = make_path([(1, 0), (1, 1)])
    trace = d.get_trace()
    assert trace.trace_v(origin, (unit_x + unit_y)) == V2(1.0, 1.0)
    assert trace.trace_v(origin, (unit_x + unit_y).normalized()) == V2(
        1.0, 1.0
    )


def test_transform() -> None:
    square = make_path([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    env = square.scale_x(2).scale_y(3).get_envelope()
    assert env(unit_x) == 2
    assert env(2 * unit_x) == 1
    assert env(unit_y) == 3
    env = square.rotate(45).get_envelope()
    assert env((unit_x + unit_y).normalized()) == pytest.approx(1)
    env = square.translate(-2, -2).get_envelope()
    assert env(unit_x) == pytest.approx(-1)
