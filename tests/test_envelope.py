import math
from chalk import *
import pytest
from hypothesis import given
from hypothesis.strategies import composite, integers, lists, one_of, sampled_from
import chalk

@composite
def vectors(draw):
    x = draw(integers(min_value=-2, max_value=2).filter(lambda x: x != 0))
    y = draw(integers(min_value=-2, max_value=2).filter(lambda x: x != 0))
    return V2(x, y)

@composite
def trails(draw):
    vs = draw(lists(vectors(), min_size=1))
    return Trail.from_offsets(vs)

@composite
def paths(draw):
    return draw(trails()).stroke().center_xy()

@composite
def circles(draw):
    return circle(draw(integers(min_value=1)))

@composite
def rects(draw):
    return rectangle(draw(integers(min_value=1)), draw(integers(min_value=1)))

@composite
def shapes(draw):
    return draw(one_of(paths(),  rects()))

@composite
def diagrams(draw):
    shape = draw(shapes())
    transform = draw(transforms())
    return shape.apply_transform(transform)

@composite
def transforms(draw):

    v2 = draw(vectors())
    return draw(sampled_from([chalk.transform.Affine.scale(v2),
                              chalk.transform.Affine.translation(v2),
                              chalk.transform.Affine.rotation(v2.angle),
                              ]))



@given(diagrams(), vectors())
def test_envelope_trail(diagram, vec):
    "Property -> Envelope bounds trace."
    trace = diagram.get_trace()
    env = diagram.get_envelope()
    ts = trace(P2(0, 0), vec)
    e = env(vec)
    for t in ts:
        assert e == pytest.approx(t) or e > t

# Some specific tests. 
def test_square():
    square = make_path([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    env = square.get_envelope()
    assert env(unit_x) == 1
    assert env(2 * unit_x) == 0.5
    assert env(unit_y) == 1
    assert env((unit_x + unit_y).normalized()) == pytest.approx(math.sqrt(1 + 1))
    
def test_circle():
    d = circle(1)
    env = d.get_envelope()
    assert env(unit_x) == 1
    assert env(2 * unit_x) == 0.5
    assert env(unit_y) == 1
    assert env((unit_x + unit_y).normalized()) == pytest.approx(1)

def test_circle_trace():
    d = circle(1)
    trace = d.get_trace()
    assert set(trace(origin, unit_x)) == set([-1.0, 1.0])
    assert set(trace(origin, (2 * unit_x))) == set([-0.5, 0.5])
    assert set(trace(origin, unit_y)) == set([-1.0, 1.0])
    # assert set(trace(origin, (unit_x + unit_y).normalized())) == pytest.approx(set([-1.0, 1.0]))

def test_path_trace():
    d = make_path([(1, 0), (1, 1)])
    trace = d.get_trace()
    assert trace.trace_v(origin, (unit_x + unit_y)) == V2(1.0, 1.0)
    assert trace.trace_v(origin, (unit_x + unit_y).normalized()) == V2(1.0, 1.0)

    
    
def test_transform():
    square = make_path([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    env = square.scale_x(2).scale_y(3).get_envelope()
    assert env(unit_x) == 2
    assert env(2 * unit_x) == 1
    assert env(unit_y) == 3
    env = square.rotate(45).get_envelope()
    assert env((unit_x + unit_y).normalized()) == pytest.approx(1)
    env = square.translate(-2, -2).get_envelope()
    assert env(unit_x) == pytest.approx(-1)



    
