from enum import Enum


class Point:
    def __init__(self, _id, coord):
        self.id = _id
        self.coord = coord
        self.dims = len(coord)


class HalfSpace:
    def __init__(self, _id, coeff, known):
        self.id = _id
        self.coeff = coeff
        self.known = known
        self.dims = len(coeff)


class Position(Enum):
    ABOVE = 1
    BELOW = -1
    OVERLAPPED = 0


def find_pointhalfspace_position(point, halfspace):
    val = sum(halfspace.coeff * point.coord)

    # The position is referred to the halfspace with respect to the point
    if val < halfspace.known:
        return Position.ABOVE
    elif val > halfspace.known:
        return Position.BELOW
    else:
        return Position.OVERLAPPED
