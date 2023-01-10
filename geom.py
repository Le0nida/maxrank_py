from enum import Enum


"""
Geom
Contains classes representing geometrical objects used in MaxRank procedures along with some useful methods.
"""



class Point:
    def __init__(self, coord, _id=None):
        self.id = _id
        self.coord = coord
        self.dims = len(coord)


class HalfLine:
    """
    Represents the score halfline relative to the point "pnt" and the target. Used only in 2D procedures.
    The halfline equation is stored in the explicit form y = m*x + q.
    """
    def __init__(self, pnt):
        self.pnt = pnt
        self.m = pnt.coord[0] - pnt.coord[1]
        self.q = pnt.coord[1]
        self.arr = Arrangement.AUGMENTED
        self.dims = 2

    def get_y(self, x):
        return self.m * x + self.q


def find_halflines_intersection(r, s):
    """
    Calculates the intersection point between two halflines.
    """

    if r.m == s.m:
        return None
    else:
        x = (s.q - r.q) / (r.m - s.m)

        return Point([x, r.get_y(x)])


class HalfSpace:
    """
    Represents the score halfspace relative to the point "pnt" and the target.
    The halfspace equation is stored as the unknowns coefficients (coeff) and the known terms (known).
    """
    def __init__(self, pnt, coeff, known):
        self.pnt = pnt
        self.coeff = coeff
        self.known = known
        self.arr = Arrangement.AUGMENTED
        self.dims = len(coeff)


def genhalfspaces(p, records):
    """
    Generates all score halfspaces relative to the target "p" and all incomparable data points "records".
    See MaxRank paper for reference on the formula.
    """

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


def find_pointhalfspace_position(point, halfspace):
    """
    Evaluates the reciprocal position between a point and an halfspace.
    """

    val = halfspace.coeff.dot(point.coord)

    if val < halfspace.known:
        return Position.IN
    elif val > halfspace.known:
        return Position.OUT
    else:
        return Position.ON


class Arrangement(Enum):
    """
    Characterizes an halfspace(line) as singular or augmented as defined in the MaxRank paper.
    Used only in AA algorithms.
    """

    SINGULAR = 0
    AUGMENTED = 1
