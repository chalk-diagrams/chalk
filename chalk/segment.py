from chalk.point import Point, Vector


@dataclass
class Segment:
    p: Point
    q: Point

    def get_trace(self) -> Trace:
        def f(point: Point, direction: Vector) -> List[SignedDistance]:
            import pdb; pdb.set_trace()

        return Trace(f)



@dataclass
class Line:
    p: Point
    v: Vector


def line_line_intersection(line1: Line, line2: Line) -> Optional[Tuple[float, float]]:
    u = line2.v - line1.v
    x1 = line1.v.cross(line2.v)
    x2 = u.cross(line1.v)
    x3 = u.cross(line2.v)
    if x1 == 0 and x2 != 0:
        return None
    else:
        return x3 / x1, x2 / x1
