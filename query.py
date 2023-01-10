import numpy as np


# TODO Implement Rtree to optimize these operations (BBS for sky)
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


def getskyline(data):
    def dominates(p, r):
        return np.all(p.coord <= r.coord) and np.any(p.coord < r.coord)

    window = []

    for pnt in data:
        dominated = False
        for w_pnt in window:
            if dominates(w_pnt, pnt):
                dominated = True
                break

        if not dominated:
            for w_pnt in reversed(window):
                if dominates(pnt, w_pnt):
                    window.remove(w_pnt)

            window.append(pnt)

    return window
