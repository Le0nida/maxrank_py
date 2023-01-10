import numpy as np


"""
Query
Utility methods for querying data

NOTE: An Rtree implementation could be used to optimize all these operations (BBS for skyline). 
Considering Python itself is a way narrower bottleneck of performance than any of these methods, the Rtree implementation was abandoned.
"""



def getdominators(data, p):
    """
    Computes the dominators of target record "p".
    """

    dominators = []
    for r in data:
        if np.all(r.coord <= p.coord) and np.any(r.coord < p.coord):
            dominators.append(r)

    return dominators


def getdominees(data, p):
    """
    Computes the dominees of target record "p".   
    """

    dominees = []
    for r in data:
        if np.all(r.coord >= p.coord) and np.any(r.coord > p.coord):
            dominees.append(r)

    return dominees


def getincomparables(data, p):
    """
    Computes the records incomparable to target record "p".   
    """

    incomp = []
    for r in data:
        if np.any(r.coord < p.coord) and np.any(r.coord > p.coord):
            incomp.append(r)

    return incomp


def getskyline(data):
    """
    Computes the skyline of "data" using BNL.
    """
    
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
