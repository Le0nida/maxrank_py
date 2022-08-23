import math
import numpy as np

import query
from geom import *
from qtree import QTree


class Cell:
    order = None
    mask = None
    covered = None
    halfspaces = None
    feasible_pnt = None

    def __init__(self):
        pass


def genhammingstrings(strlen, weight):
    if weight == 0:
        return [np.binary_repr(0, width=strlen)]
    elif weight == 1:
        decstr = [2 ** b for b in range(strlen)]
        return [np.binary_repr(decstr[i], width=strlen) for i in range(len(decstr))]
    else:
        halfmax = 2 ** (strlen - 1) - 1
        curr_weight = 2

        decstr = [2 ** b + 1 for b in range(1, strlen)]
        bases = [decstr[i] for i in range(len(decstr)) if decstr[i] <= halfmax]

        while True:
            while len(bases) > 0:
                shifts = np.left_shift(bases, 1)
                decstr = decstr + list(shifts)
                bases = [shifts[i] for i in range(len(shifts)) if shifts[i] <= halfmax]

            if curr_weight < weight:
                decstr = [2 * decstr[i] + 1 for i in range(len(decstr)) if 2 * decstr[i] + 1 <= 2 ** strlen - 1]
                bases = [decstr[i] for i in range(len(decstr)) if decstr[i] <= halfmax]
                curr_weight += 1
            else:
                break

        return [np.binary_repr(decstr[i], width=strlen) for i in range(len(decstr))]


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
                if hamstr[b] == '0':
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
                cell.feasible_pnt = point
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
                print("> Leaf {}: Evaluating Hamming strings of weight {}".format(id(leaf), hamweight))
            hamstrings = genhammingstrings(len(leaf.halfspaces), hamweight)
            cells = searchmincells(leaf.mbr, hamstrings, leaf.halfspaces)

            if len(cells) > 0:
                for cell in cells:
                    cell.order = leaf.order + hamweight

                if minorder > leaf.order + hamweight:
                    minorder = leaf.order + hamweight
                    mincells = cells
                    print("> Leaf {}: Found {} mincell(s) with a minorder of {}".format(id(leaf), len(mincells), minorder))
                else:
                    mincells = mincells + cells
                    print("> Leaf {}: Found another {} mincell(s)".format(id(leaf), len(cells)))
                break

            hamweight += 1

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
    n_exp = 0

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
                    print("> Leaf {}: Evaluating Hamming strings of weight {}".format(id(leaf), hamweight))
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

        print("> Expansion {}: Found {} mincell(s)".format(n_exp, len(mincells)))

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
                print("> Expansion {}: Found a singular mincell(s) with a minorder of {}".format(n_exp, minorder_singular))

        if len(to_expand) == 0:
            break
        else:
            print("> Expansion {}: {} halfspace(s) will be expanded".format(n_exp, len(to_expand)))
            for hs in to_expand:
                hs.arr = Arrangement.SINGULAR
                incomp.remove(hs.pnt)

            new_sky = query.getskyline(incomp)
            new_halfspaces = genhalfspaces(p, [pnt for pnt in new_sky if pnt not in sky])

            for hs in new_halfspaces:
                qt.inserthalfspace(qt.root, hs)
            if len(new_halfspaces) > 0:
                print("> {} new halfspace(s) have been inserted".format(len(new_halfspaces)))

            sky = new_sky

            leaves = qt.getleaves()
            for leaf in leaves:
                leaf.getorder()
            leaves.sort(key=lambda x: x.order)

            n_exp += 1

    return len(dominators) + minorder_singular + 1, mincells_singular
