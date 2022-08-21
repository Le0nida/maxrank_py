import numpy as np

from geom import HalfSpace


def getdominators(data, p):
    dominators = []
    for r in data:
        if np.all(r.coord <= p.coord) and np.any(r.coord < p.coord):
            dominators.append(r)

    return dominators


def getdominees(data, p):
    dominees = []
    for r in data:
        if np.all(r.coord >= p.coord) and np.any(r.coord > p.coord):
            dominees.append(r)

    return dominees


def getincomparables(data, p):
    incomp = []
    for r in data:
        if np.any(r.coord < p.coord) and np.any(r.coord > p.coord):
            incomp.append(r)

    return incomp


def genhalfspaces(p, records):
    halfspaces = []
    p_d = p.coord[-1]
    p_i = p.coord[:-1]

    for r in records:
        r_d = r.coord[-1]
        r_i = r.coord[:-1]

        halfspaces.append(HalfSpace(r.id, r_i - r_d - p_i + p_d, p_d - r_d))

    return halfspaces
