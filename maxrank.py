import numpy as np
from sortedcontainers import SortedList
from scipy.optimize import linprog
import ctypes

import query
from geom import *
from qtree import QTree

"""
Maxrank
Implements the actual MaxRank procedures.
"""

lib = ctypes.CDLL('./lpsolver.so')


# Definisci la struttura del risultato
class LinprogResult(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.POINTER(ctypes.c_double)),
        ("fun", ctypes.c_double),
        ("status", ctypes.c_int),
        ("message", ctypes.c_char * 128),
    ]

# Definisci le firme delle funzioni
lib.linprog_highs.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # c
    ctypes.POINTER(ctypes.c_double),  # A_ub
    ctypes.POINTER(ctypes.c_double),  # b_ub
    ctypes.POINTER(ctypes.c_double),  # bounds
    ctypes.c_int,  # num_vars
    ctypes.c_int,  # num_constraints
]
lib.linprog_highs.restype = ctypes.POINTER(LinprogResult)
lib.free_linprog_result.argtypes = [ctypes.POINTER(LinprogResult)]

# Funzione per chiamare il solutore
def linprog_highs(c, A_ub, b_ub, bounds):
    num_vars = len(c)
    num_constraints = len(b_ub)

    c = (ctypes.c_double * num_vars)(*c)
    A_ub = (ctypes.c_double * (num_vars * num_constraints))(*A_ub.flatten())
    b_ub = (ctypes.c_double * num_constraints)(*b_ub)

    # Sostituisci None con np.inf per i bounds
    bounds_array = []
    for lb, ub in bounds:
        if lb is None:
            lb = -np.inf
        if ub is None:
            ub = np.inf
        bounds_array.append(lb)
        bounds_array.append(ub)

    bounds = (ctypes.c_double * (2 * num_vars))(*bounds_array)

    result = lib.linprog_highs(c, A_ub, b_ub, bounds, num_vars, num_constraints)
    solution = np.array([result.contents.x[i] for i in range(num_vars)])
    fun = result.contents.fun
    status = result.contents.status
    message = result.contents.message.decode()

    lib.free_linprog_result(result)

    return solution, fun, status, message


# Test della funzione
if __name__ == "__main__":
    c = np.array([0.0, 0.0, -1.0])
    A_ub = np.array([
        [-0.61385003, -0.36678181,  1.0],
        [ 0.82523785,  0.32724379,  1.0],
        [-1.16851646, -0.63505809,  1.0],
        [ 1.0,         1.0,         0.0]
    ])
    b_ub = np.array([-0.53232016, 0.68554353, -0.72504181, 1.0])
    bounds = [(0.0, 0.5), (0.5, 1.0), (0, None)]

    solution, fun, status, message = linprog_highs(c, A_ub, b_ub, bounds)
    print("\n\nSolution:", solution)
    print("Objective value:", fun)
    print("Status:", status)
    print("Message:", message)

class Cell:
    """
    Object class representing a Mincell.
    """

    def __init__(self, order, mask, covered, halfspaces, leaf_mbr, feasible_pnt):
        self.order = order
        self.mask = mask
        self.covered = covered
        self.halfspaces = halfspaces
        self.leaf_mbr = leaf_mbr
        self.feasible_pnt = feasible_pnt

    def issingular(self):
        return all([hs.arr == Arrangement.SINGULAR for hs in self.covered])


class Interval:
    """
    Object class representing a 1D Mincell.
    """

    order = None
    covered = []

    def __init__(self, halfline, _range, coversleft):
        self.halfline = halfline
        self.range = _range
        self.coversleft = coversleft

    def issingular(self):
        return all([hl.arr == Arrangement.SINGULAR for hl in self.covered])


def genhammingstrings(strlen, weight):
    """
    Builds all Hamming strings given a length and a weight (number of 1s).
    The approach used is bottom-up, building strings with a bigger weight from those with a lower one. The strings are handled in their binary rep and ultimately converted.
    """

    if weight > np.ceil(strlen / 2):
        weight = strlen - weight
        botup = False
    else:
        botup = True

    if weight == 0:
        decstr = [0]
    elif weight == 1:
        decstr = [2 ** b for b in range(strlen)]
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

    if botup:
        return [np.binary_repr(decstr[i], width=strlen) for i in range(len(decstr))]
    else:
        decmax = 2 ** strlen - 1
        return [np.binary_repr(decmax - decstr[i], width=strlen) for i in range(len(decstr))]


def print_array(arr, name):
    print(f"{name}:")
    print(arr)
    print("\n\n")


def searchmincells_lp(leaf, hamstrings):
    """
    Mincell search algorithm using linear programming.

    max x_d+1
    s.t.
        sum(c_ij * x_ij) + x_d+1 <= d_j     for i = 1,...,dims and for j=1,...,len(halfspaces)
        sum(x_i) <= 1                       for i = 1,...,dims
        x_i in leaf.mbr[i, :]               for i = 1,...,dims
        x_d+1 in [0, +inf]

    > The first (set of) constraint(s) define(s) the halfspaces arrangment according to the hamstring
    > The second contraint is the normalization bound: all query parameters must sum to 1
    > The third constraint states that the query must fall into the current leaf mbr
    > Finally the last constraint forces the "slack" to be positive
    """

    cells = []
    dims = leaf.mbr.shape[0]
    leaf_covered = leaf.getcovered()

    # If there are no halfspaces, then the whole leaf is the mincell
    if len(leaf.halfspaces) == 0:
        return [Cell(
            None,
            None,
            leaf_covered,
            [],
            leaf.mbr,
            Point(np.random.uniform(low=leaf.mbr[:, 0], high=leaf.mbr[:, 1], size=dims))
        )]

    c = np.zeros(dims + 1)
    c[-1] = -1

    # Create A_ub and b_ub
    A_ub = np.ones((len(leaf.halfspaces) + 1, dims + 1))
    A_ub[-1, -1] = 0

    b_ub = np.ones(len(leaf.halfspaces) + 1)

    bounds = [(leaf.mbr[d, 0], leaf.mbr[d, 1]) for d in range(dims)]
    bounds += [(0, None)]

    for hamstr in hamstrings:
        for b in range(len(hamstr)):
            if hamstr[b] == '0':
                A_ub[b, :-1] = -leaf.halfspaces[b].coeff
                b_ub[b] = -leaf.halfspaces[b].known
            else:
                A_ub[b, :-1] = leaf.halfspaces[b].coeff
                b_ub[b] = leaf.halfspaces[b].known

        print("\n\n\n")

        # Print array data before conversion to ctypes
        print_array(c, "Objective coefficients (c)")
        print_array(A_ub, "Constraint coefficients (A_ub)")
        print_array(b_ub, "Constraint bounds (b_ub)")
        print_array(bounds, "Variable bounds")

        print("\n\n\n")

        solution, fun, status, message = linprog_highs(c, A_ub, b_ub, bounds)
        print("Solution:", solution)
        print("Objective value:", fun)
        print("Status:", status)
        print("Message:", message)

        success = False
        if success:
            cell = Cell(
                None,
                hamstr,
                leaf_covered + [leaf.halfspaces[b] for b in range(len(hamstr)) if hamstr[b] == '1'],
                leaf.halfspaces,
                leaf.mbr,
                Point(x[:-1])
            )
            cells.append(cell)
            break

    return cells


def ba_hd(data, p):
    """
    Basic Approach to compute MaxRank when DIM > 2 as described in the MaxRank paper.
    """

    qt = QTree(p.dims - 1, 10)

    dominators = query.getdominators(data, p)
    incomp = query.getincomparables(data, p)

    halfspaces = genhalfspaces(p, incomp)

    if len(halfspaces) > 0:
        qt.inserthalfspace(halfspaces)
    print("> {} halfspaces have been inserted".format(len(halfspaces)))

    leaves = qt.getleaves()
    for leaf in leaves:
        leaf.getorder()
    leaves.sort(key=lambda x: x.order)

    minorder = np.inf
    mincells = []
    for leaf in leaves:
        if leaf.order > minorder:
            break

        hamweight = 0
        while hamweight <= len(leaf.halfspaces) and leaf.order + hamweight <= minorder:
            if hamweight >= 3:
                print("> Leaf {}: Evaluating Hamming strings of weight {}".format(id(leaf), hamweight))
            hamstrings = genhammingstrings(len(leaf.halfspaces), hamweight)
            cells = searchmincells_lp(leaf, hamstrings)

            if len(cells) > 0:
                for cell in cells:
                    cell.order = leaf.order + hamweight

                if minorder > leaf.order + hamweight:
                    minorder = leaf.order + hamweight
                    mincells = cells
                    print("> Leaf {}: Found {} mincell(s) with a minorder of {}".format(id(leaf), len(mincells),
                                                                                        minorder))
                else:
                    mincells = mincells + cells
                    print("> Leaf {}: Found another {} mincell(s)".format(id(leaf), len(cells)))
                break

            hamweight += 1

    return len(dominators) + minorder + 1, mincells


def aa_2d(data, p):
    """
    Advanced Approach to compute MaxRank when DIM = 2 as described in the MaxRank paper.
    """

    # Compute dominators and incomparables
    dominators = query.getdominators(data, p)
    incomp = query.getincomparables(data, p)
    sky = query.getskyline(incomp)

    p_line = HalfLine(p)

    ints = SortedList(key=lambda el: el.range[1])
    for sp in sky:
        sp_line = HalfLine(sp)
        pnt = find_halflines_intersection(p_line, sp_line)
        ints.add(Interval(sp_line, np.array([np.nan, pnt.coord[0]]), sp_line.q < p_line.q))
    ints.add(Interval(None, np.array([np.nan, 1]), False))
    print("> {} halflines(s) have been inserted".format(len(sky)))

    n_exp = 0

    while True:
        minorder = np.inf
        mincells = []
        last_end = 0

        covering = [cell.halfline for cell in ints if cell.coversleft]

        for cell in ints:
            cell.order = len(covering)
            cell.covered = covering.copy()
            cell.range[0] = last_end
            last_end = cell.range[1]

            if cell.order < minorder:
                minorder = cell.order
                mincells = [cell]
            elif cell.order == minorder:
                mincells += [cell]

            if cell.coversleft:
                covering.pop(0)
            else:
                covering.append(cell.halfline)
        print("> Expansion {}: Found {} mincell(s)".format(n_exp, len(mincells)))

        # Check all mincells found for singulars; if they aren't put their halflines up for expansion
        mincells_singular = []
        to_expand = []
        for cell in mincells:
            if cell.issingular():
                mincells_singular.append(cell)
            else:
                to_expand += [hl for hl in cell.covered if hl.arr == Arrangement.AUGMENTED and hl not in to_expand]
        if len(mincells_singular) > 0:
            print("> Expansion {}: Found {} singular mincell(s) with a minorder of {}"
                  .format(n_exp, len(mincells_singular), minorder))

        # If there aren't any new halflines to expand then the search is terminated
        if len(to_expand) == 0:
            return len(dominators) + minorder + 1, mincells_singular
        else:
            n_exp += 1

            print("> Expansion {}: {} halfline(s) will be expanded".format(n_exp, len(to_expand)))
            for hl in to_expand:
                hl.arr = Arrangement.SINGULAR
                incomp.remove(hl.pnt)

            new_sky = query.getskyline(incomp)
            to_insert = [sp for sp in new_sky if sp not in sky]
            sky = new_sky

            for sp in to_insert:
                sp_line = HalfLine(sp)
                pnt = find_halflines_intersection(p_line, sp_line)
                ints.add(Interval(sp_line, np.array([np.nan, pnt.coord[0]]), sp_line.q < p_line.q))
            if len(to_insert) > 0:
                print("> {} halflines(s) have been inserted".format(len(to_insert)))


def aa_hd(data, p):
    """
    Advanced Approach to compute MaxRank when DIM > 2 as described in the MaxRank paper.
    """

    def updateqt(old_sky):
        """
        Computes skyline of incomparables, insert their halfspaces in the QTree and retrieves the leaves.
        """

        new_sky = query.getskyline(incomp)
        new_halfspaces = genhalfspaces(p, [pnt for pnt in new_sky if pnt not in old_sky])

        if len(new_halfspaces) > 0:
            qt.inserthalfspace(new_halfspaces)
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
            while hamweight <= len(leaf.halfspaces) \
                    and leaf.order + hamweight <= minorder \
                    and leaf.order + hamweight <= minorder_singular:
                if hamweight >= 3:
                    print("> Leaf {}: Evaluating Hamming strings of weight {}".format(id(leaf), hamweight))
                hamstrings = genhammingstrings(len(leaf.halfspaces), hamweight)
                cells = searchmincells_lp(leaf, hamstrings)

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
        new_singulars = 0
        to_expand = []
        for cell in mincells:
            if cell.issingular():
                minorder_singular = cell.order
                mincells_singular.append(cell)
                new_singulars += 1
            else:
                to_expand += [hs for hs in cell.covered if hs.arr == Arrangement.AUGMENTED and hs not in to_expand]
        if new_singulars > 0:
            print("> Expansion {}: Found {} singular mincell(s) with a minorder of {}"
                  .format(n_exp, new_singulars, minorder_singular))

        # If there aren't any new halfspaces to expand then the search is terminated
        if len(to_expand) == 0:
            return len(dominators) + minorder_singular + 1, mincells_singular
        else:
            # Otherwise, remove the correspondent incomparables and update the QTree
            n_exp += 1

            print("> Expansion {}: {} halfspace(s) will be expanded".format(n_exp, len(to_expand)))
            for hs in to_expand:
                hs.arr = Arrangement.SINGULAR
                incomp.remove(hs.pnt)

            sky, leaves = updateqt(sky)
