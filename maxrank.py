import numpy as np

import query
from geom import *
from qtree import QTree


class Cell:
    order = None
    mask = None
    halfspaces = None

    def __init__(self):
        pass


def genhammingstrings(strlen, weight):
    def advance_counter(pos):
        counter[pos] += 1
        if counter[pos] > strlen + pos - weight and pos > 0:
            counter[pos] = advance_counter(pos - 1) + 1
            return counter[pos]
        else:
            return counter[pos]

    if weight == 0:
        return np.zeros(strlen).reshape(1, strlen)
    if weight == 1:
        return np.eye(strlen)
    else:
        counter = np.arange(weight)

        hamstr = np.zeros((1, strlen))
        for c in counter:
            hamstr[0, c] = 1
        advance_counter(weight - 1)

        while counter[0] <= strlen - weight:
            newstr = np.zeros((1, strlen))
            for c in counter:
                newstr[0, c] = 1
            hamstr = np.append(hamstr, newstr, axis=0)

            advance_counter(weight - 1)
        return hamstr


def searchmincells(mbr, hamstrings, halfspaces):
    cells = []

    if len(halfspaces) == 0:
        cell = Cell()
        cell.mask = []
        # TODO put quadrant mbr as halfspaces
        cells.append(cell)
        return cells

    for hamstr in hamstrings:
        for i in range(5000):
            found = True
            point = Point(None, np.random.uniform(low=mbr[:, 0], high=mbr[:, 1], size=halfspaces[0].dims))

            for b in range(len(hamstr)):
                if hamstr[b] == 0:
                    if not find_pointhalfspace_position(point, halfspaces[b]) == Position.BELOW:
                        found = False
                        break
                else:
                    if not find_pointhalfspace_position(point, halfspaces[b]) == Position.ABOVE:
                        found = False
                        break
            if found:
                cell = Cell()
                cell.mask = hamstr
                cell.halfspaces = halfspaces
                cells.append(cell)
                break
    return cells


def ba_hd(data, p):
    qt = QTree(p.dims - 1, 10)

    dominators = query.getdominators(data, p)
    incomp = query.getincomparables(data, p)

    halfspaces = query.genhalfspaces(p, incomp)

    for hs in halfspaces:
        qt.inserthalfspace(qt.root, hs)
    print("> {} halfspaces have been inserted".format(len(halfspaces)))

    leaves = qt.getleaves()
    for leaf in leaves:
        leaf.getorder()

    leaves.sort(key=lambda x: x.order)

    minorder = np.inf
    mincells = []
    niter = 0
    for leaf in leaves:
        if leaf.order > minorder:
            break

        hamweight = 0
        while hamweight <= len(leaf.halfspaces) or leaf.order + hamweight > minorder:
            hamstrings = genhammingstrings(len(leaf.halfspaces), hamweight)
            cells = searchmincells(leaf.mbr, hamstrings, leaf.halfspaces)

            if len(cells) > 0:
                for cell in cells:
                    cell.order = leaf.order + hamweight

                if minorder > leaf.order + hamweight:
                    minorder = leaf.order + hamweight
                    mincells = cells
                else:
                    mincells = mincells + cells
                break

            hamweight += 1

    niter += 1
    if niter % 5 == 0:
        print("> {} leaves have been processed...".format(niter))

    return minorder + len(dominators), mincells
