from chalk.arc import ArcBetween, ArcByDegrees, ArcRadius, Degrees
from chalk.transform import P2

from hypothesis import given
from hypothesis.strategies import floats
import pytest

def test_arc_between():
    a = ArcByDegrees(Degrees(180), Degrees(0), 1, P2(0, 0))
    b = ArcBetween(P2(-1, 0), P2(1,0), 1)
    c = ArcRadius(P2(-1, 0), P2(1,0), 0, 1)

    assert a.to_arc_between() == b
    assert b.to_arc_by_degrees() == a
    assert c.to_arc_between() == b
    assert b.to_arc_radius() == c


degree = floats(min_value=10, max_value=350)
@given(degree, degree)
def test_arcs(angle0, angle_off):
    a = ArcByDegrees(Degrees(angle0), Degrees(angle0 - angle_off), 10, P2(0, 0))
    b = a.to_arc_between().to_arc_by_degrees()
    print(a, a.to_arc_between(), b)
    assert a.radius == pytest.approx(b.radius)
    # assert a.angle0 == pytest.approx(b.angle0)
    # assert a.angle1 == pytest.approx(b.angle1)
