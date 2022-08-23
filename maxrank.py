import numpy as np

import query
from geom import *
from qtree import QTree


class Cell:
    order = None
    mask = None
    covered = None
    halfspaces = None

    def __init__(self):
        pass


# TODO Implement smarter generation using shifts and bottom-up construction
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


# TODO Convert from MonteCarlo to Linear Programming
def searchmincells(mbr, hamstrings, halfspaces):
    cells = []

    if len(halfspaces) == 0:
        cell = Cell()
        cell.mask = []
        # TODO Put quadrant mbr as halfspaces
        cells.append(cell)
        return cells

    for hamstr in hamstrings:
        for i in range(5000):
            found = True
            while True:
                point = Point(None, np.random.uniform(low=mbr[:, 0], high=mbr[:, 1], size=halfspaces[0].dims))
                if sum(point.coord) <= 1:
                    break

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

    halfspaces = genhalfspaces(p, incomp)

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
        while hamweight <= len(leaf.halfspaces) and leaf.order + hamweight <= minorder:
            if hamweight >= 2:
                print("> Evaluating Hamming strings of weight {}".format(hamweight))
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

    return len(dominators) + minorder + 1, mincells


def aa_hd(data, p):
    qt = QTree(p.dims - 1, 10)

    dominators = query.getdominators(data, p)
    incomp = query.getincomparables(data, p)

    sky = query.getskyline(incomp)
    halfspaces = genhalfspaces(p, sky)

    for hs in halfspaces:
        qt.inserthalfspace(qt.root, hs)
    print("> {} halfspaces have been inserted".format(len(halfspaces)))

    leaves = qt.getleaves()
    for leaf in leaves:
        leaf.getorder()
    leaves.sort(key=lambda x: x.order)

    minorder_singular = np.inf
    mincells_singular = []

    while True:
        minorder = np.inf
        mincells = []
        for leaf in leaves:
            if leaf.order > minorder or leaf.order > minorder_singular:
                break

            hamweight = 0
            while hamweight <= len(leaf.halfspaces) \
                    and leaf.order + hamweight <= minorder \
                    and leaf.order + hamweight <= minorder_singular:
                if hamweight >= 2:
                    print("> Evaluating Hamming strings of weight {}".format(hamweight))
                hamstrings = genhammingstrings(len(leaf.halfspaces), hamweight)
                cells = searchmincells(leaf.mbr, hamstrings, leaf.halfspaces)

                if len(cells) > 0:
                    leaf_covered = []
                    ref = leaf.parent
                    while not ref.isroot():
                        leaf_covered = leaf_covered + ref.covered
                        ref = ref.parent

                    for cell in cells:
                        cell.order = leaf.order + hamweight

                        cell.covered = []
                        for b in range(len(cell.mask)):
                            if cell.mask[b] == 1:
                                cell.covered.append(cell.halfspaces[b])
                        cell.covered = cell.covered + leaf_covered

                    if minorder > leaf.order + hamweight:
                        minorder = leaf.order + hamweight
                        mincells = cells
                    else:
                        mincells = mincells + cells
                    break

                hamweight += 1

        to_expand = []
        for cell in mincells:
            singular = True
            for hs in cell.covered:
                if not hs.arr == Arrangement.SINGULAR:
                    singular = False
                    if hs not in to_expand:
                        to_expand.append(hs)
            if singular:
                minorder_singular = cell.order
                mincells_singular.append(cell)

        if len(to_expand) == 0:
            break
        else:
            for hs in to_expand:
                hs.arr = Arrangement.SINGULAR
                incomp.remove(hs.pnt)

            new_sky = query.getskyline(incomp)
            new_halfspaces = genhalfspaces(p, [pnt for pnt in new_sky if pnt not in sky])

            for hs in new_halfspaces:
                qt.inserthalfspace(qt.root, hs)
            if len(new_halfspaces) > 0:
                print("> {} new halfspaces have been inserted".format(len(new_halfspaces)))

            sky = new_sky

            leaves = qt.getleaves()
            for leaf in leaves:
                leaf.getorder()
            leaves.sort(key=lambda x: x.order)

    return len(dominators) + minorder_singular + 1, mincells_singular
