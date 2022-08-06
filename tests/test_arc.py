from chalk.arc import ArcSegment
from chalk.transform import P2, V2

from hypothesis import given
from hypothesis.strategies import floats, booleans
import pytest
import math

def test_arc_between():
    checks = []
    
    a = ArcSegment(180, 180)
    b = ArcSegment.arc_between(P2(-1, 0), P2(1, 0), 1)
    checks.append((a, b))

    a = ArcSegment(180, -180)
    b = ArcSegment.arc_between(P2(-1, 0), P2(1, 0), -1)
    checks.append((a, b))
    
    for a, b in checks:
        assert a.p == b.p
        assert a.q == b.q
        assert a.center == b.center


def test_arc_envelope():
    checks = []
    
    a = ArcSegment(180, 180)
    assert a.get_envelope()(V2(0, -1)) == 1
    assert a.get_envelope()(V2(0, 1)) == 0


    a = ArcSegment(180, 180).scale_y(0.5)
    assert a.get_envelope()(V2(0, -1)) == 0.5
    assert a.get_envelope()(V2(0, 1)) == 0
    
