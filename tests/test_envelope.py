import math
from chalk import *
import pytest

def test_square():
    square = make_path([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    env = square.get_envelope()
    assert env(V2(1, 0)) == 1
    assert env(V2(2, 0)) == 2
    assert env(V2(0, 1)) == 1
    assert env(V2(1, 1).normalized()) == pytest.approx(math.sqrt(1 + 1))
    
def test_circle():
    d = circle(1)
    env = d.get_envelope()
    assert env(V2(1, 0)) == 1
    assert env(V2(2, 0)) == 2
    assert env(V2(0, 1)) == 1
    assert env(V2(1, 1).normalized()) == pytest.approx(1)
    
def test_transform():
    square = make_path([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    env = square.scale_x(2).scale_y(3).get_envelope()
    assert env(V2(1, 0)) == 2
    assert env(V2(2, 0)) == 4
    assert env(V2(0, 1)) == 3
    env = square.rotate(45).get_envelope()
    assert env(V2(1, 1).normalized()) == pytest.approx(1)
    env = square.translate(-2, -2).get_envelope()
    assert env(V2(1, 0)) == pytest.approx(-1)

    
