from enum import Enum


class Point:
    def __init__(self, coord, _id=None):
        self.id = _id
        self.coord = coord
        self.dims = len(coord)


class HalfSpace:
    def __init__(self, pnt, coeff, known):
        self.pnt = pnt
        self.coeff = coeff
        self.known = known
        self.arr = Arrangement.AUGMENTED
        self.dims = len(coeff)


class HalfLine:
    def __init__(self, pnt):
        self.pnt = pnt
        self.m = pnt.coord[0] - pnt.coord[1]
        self.q = pnt.coord[1]
        self.arr = Arrangement.AUGMENTED
        self.dims = 2

    def get_y(self, x):
        return self.m * x + self.q


class Position(Enum):
    """
    Defines the reciprocal position between a point and a halfspace.
    A point can be:
        IN -> Inside the halfspace: satisfies the halfspace disequation
        OUT -> Outside the halfspace: is inside the halfspace complement
        ON -> Lies on the halfspace boundary: satisfies the halfspace equation
    """
    IN = 1
    OUT = -1
    ON = 0


class Arrangement(Enum):
    SINGULAR = 0
    AUGMENTED = 1


# TODO Put this as class method
def genhalfspaces(p, records):
    halfspaces = []
    p_d = p.coord[-1]
    p_i = p.coord[:-1]

    for r in records:
        r_d = r.coord[-1]
        r_i = r.coord[:-1]

        # less-than form
        # s(r) <= s(p)
        halfspaces.append(HalfSpace(r, r_i - r_d - p_i + p_d, p_d - r_d))

    return halfspaces


# TODO Put this as class method
def find_pointhalfspace_position(point, halfspace):
    val = halfspace.coeff.dot(point.coord)

    if val < halfspace.known:
        return Position.IN
    elif val > halfspace.known:
        return Position.OUT
    else:
        return Position.ON


def find_halflines_intersection(r, s):
    if r.m == s.m:
        return None
    else:
        x = (s.q - r.q) / (r.m - s.m)

        return Point([x, r.get_y(x)])
