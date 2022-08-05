from chalk.arc import ArcBetween, ArcByDegrees, Degrees
from chalk.transform import P2

from hypothesis import given
from hypothesis.strategies import floats, booleans
import pytest
import math

def test_arc_between():
    checks = []
    
    a = ArcByDegrees(Degrees(180), Degrees(180), 1, P2(0, 0))
    b = ArcBetween(P2(-1, 0), P2(1, 0), 1)
    checks.append((a, b))

    a = ArcByDegrees(Degrees(180), Degrees(-180), 1, P2(0, 0))
    b = ArcBetween(P2(-1, 0), P2(1,0), -1)
    checks.append((a, b))
    
    for a, b in checks:
        assert a.to_arc_between() == b
        print(b, b.to_arc_by_degrees() )
        assert b.to_arc_by_degrees() == a

degree = floats(min_value=10, max_value=150)
@given(degree, degree, floats(min_value=0.1, max_value=10),
       floats(min_value=-10, max_value=10), floats(min_value=-10, max_value=10),
       booleans())
def test_arcs(angle0, angle_off, rad, p1, p2, clockwise):

    a = ArcByDegrees(Degrees(angle0), Degrees(angle_off), rad, P2(p1, p2))
    b = a.to_arc_between().to_arc_by_degrees()
    print(a)
    print(a.to_arc_between())
    print(b)
    print()
    assert a.radius == pytest.approx(b.radius)
    assert a.angle == pytest.approx(b.angle)
    assert a.dangle == pytest.approx(b.dangle)


    

