import math
import numpy as np

import query
from geom import *
from qtree import QTree


class Cell:
    def __init__(self, order, mask, covered, halfspaces, leaf_mbr, feasible_pnt):
        self.order = order
        self.mask = mask
        self.covered = covered
        self.halfspaces = halfspaces
        self.leaf_mbr = leaf_mbr
        self.feasible_pnt = feasible_pnt

    def issingular(self):
        return all([hs.arr == Arrangement.SINGULAR for hs in self.covered])


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
def searchmincells(leaf, hamstrings):
    cells = []

    # If there are no halfspaces, then the whole leaf is the mincell
    if len(leaf.halfspaces) == 0:
        return [Cell(
            None,
            None,
            leaf.covered,
            [],
            leaf.mbr,
            Point(None, np.random.uniform(low=leaf.mbr[:, 0], high=leaf.mbr[:, 1], size=leaf.halfspaces[0].dims))
        )]

    for hamstr in hamstrings:
        # MonteCarlo -> If we cen't generate a feasible point in 5000 iterations, "probably" the cell does not exist
        for i in range(5000):
            found = True
            while True:
                point = Point(None,
                              np.random.uniform(low=leaf.mbr[:, 0], high=leaf.mbr[:, 1], size=leaf.halfspaces[0].dims))
                # Only generate query points that are normalized
                if sum(point.coord) <= 1:
                    break

            # Check if the point falls in the halfspaces arrangment dictated by the hamming string
            for b in range(len(hamstr)):
                if hamstr[b] == '0':
                    if not find_pointhalfspace_position(point, leaf.halfspaces[b]) == Position.BELOW:
                        found = False
                        break
                else:
                    if not find_pointhalfspace_position(point, leaf.halfspaces[b]) == Position.ABOVE:
                        found = False
                        break

            # If the points respects all equations, that means the relative mincell exists
            if found:
                cell = Cell(
                    None,
                    hamstr,
                    leaf.covered + [leaf.halfspaces[b] for b in range(len(hamstr)) if hamstr[b] == '1'],
                    leaf.halfspaces,
                    leaf.mbr,
                    point
                )
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
            cells = searchmincells(leaf, hamstrings)

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
    # Computes skyline of incomparables, insert their halfspaces in the QTree and retrieves the leaves
    def updateqt(old_sky):
        new_sky = query.getskyline(incomp)
        new_halfspaces = genhalfspaces(p, [pnt for pnt in new_sky if pnt not in old_sky])

        for hs in new_halfspaces:
            qt.inserthalfspace(qt.root, hs)
        if len(new_halfspaces) > 0:
            print("> {} halfspace(s) have been inserted".format(len(new_halfspaces)))

        new_leaves = qt.getleaves()
        for _leaf in new_leaves:
            _leaf.getorder()
        new_leaves.sort(key=lambda x: x.order)

        return new_sky, new_leaves

    # Initialize the QTree
    qt = QTree(p.dims - 1, 10)

    # Compute dominators and incomparables
    dominators = query.getdominators(data, p)
    incomp = query.getincomparables(data, p)

    sky, leaves = updateqt([])

    minorder_singular = np.inf
    mincells_singular = []
    n_exp = 0

    # Start AA routine
    while True:
        minorder = np.inf
        mincells = []

        # Find mincells with current halfspaces, like in BA
        for leaf in leaves:
            if leaf.order > minorder or leaf.order > minorder_singular:
                break

            hamweight = 0
            while hamweight <= len(
                    leaf.halfspaces) and leaf.order + hamweight <= minorder and leaf.order + hamweight <= minorder_singular:
                if hamweight >= 2:
                    print("> Leaf {}: Evaluating Hamming strings of weight {}".format(id(leaf), hamweight))
                hamstrings = genhammingstrings(len(leaf.halfspaces), hamweight)
                cells = searchmincells(leaf, hamstrings)

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
        print("> Expansion {}: Found {} mincell(s)".format(n_exp, len(mincells)))

        # Check all mincells found for singulars; if they aren't put their halfspaces up for expansion
        to_expand = []
        for cell in mincells:
            if cell.issingular():
                minorder_singular = cell.order
                mincells_singular.append(cell)
                print("> Expansion {}: Found a singular mincell(s) with a minorder of {}".format(n_exp, minorder_singular))
            else:
                to_expand += [hs for hs in cell.covered if hs.arr == Arrangement.AUGMENTED and hs not in to_expand]

        # If there aren't any new halfspaces to expand then the search is terminated
        if len(to_expand) == 0:
            break
        else:
            # Otherwise, remove the correspondent incomparables and update the QTree
            n_exp += 1

            print("> Expansion {}: {} halfspace(s) will be expanded".format(n_exp, len(to_expand)))
            for hs in to_expand:
                hs.arr = Arrangement.SINGULAR
                incomp.remove(hs.pnt)

            sky, leaves = updateqt(sky)

    return len(dominators) + minorder_singular + 1, mincells_singular
