import numpy as np

from geom import *


"""
Qtree
Class objects to build a Quad tree and related methods.
"""



class QTree:
    """
    Main object class
    Maintain a reference to the root node and implements method to insert halfspaces and expand the tree.
    """

    def __init__(self, dims, maxhsnode):
        self.dims = dims                # Dimensionality of the space wrapped by the tree (one less than the data dimensionality)
        self.maxhsnode = maxhsnode      # Maximum number of halfspaces a node can contain before being split up
        self.masks = (genmasks(dims))   # Masks used in halfspace insertion
        self.root = self.createroot()   # Reference to root node

    def createroot(self):
        root = QNode(None, np.column_stack((np.zeros(self.dims), np.ones(self.dims))))
        self.splitnode(root)

        return root

    def splitnode(self, node):
        """
        Spiltting node routine. Called when a node contains a number of halfsapces > maxhsnode.
        """

        mindim = node.mbr[:, 0]
        maxdim = node.mbr[:, 1]

        # The number of quadrants depends on the dimensionality
        for quad in range(2 ** self.dims):
            # Convert the quadrant number in binary
            qbin = np.array(list(np.binary_repr(quad, width=self.dims)))

            # Compute new mbr
            child_mindim = np.where(qbin == '0', mindim, (mindim + maxdim) / 2)
            child_maxdim = np.where(qbin == '1', maxdim, (mindim + maxdim) / 2)

            child = QNode(node, np.column_stack((child_mindim, child_maxdim)))

            # Mark nodes lying above q1 + q2 + ... + qd = 1
            if sum(child_mindim) >= 1:
                child.norm = False

            node.children.append(child)

    def inserthalfspace(self, halfspaces):
        """
        Halfspaces insertion routine. Non-recursive.
        """

        to_search = [self.root]
        self.root.halfspaces = halfspaces

        while len(to_search) > 0:
            current = to_search.pop()

            # Calculate which of the current node's children contains which halfspace
            current.inserthalfspaces(self.masks, current.halfspaces)
            current.halfspaces = []

            for child in current.children:
                if child.norm:                                              # If the child lie outside of the normalized portion of the query space, ignore it.
                    if not child.isleaf() and len(child.halfspaces) > 0:    # If the child has children itself, the insertion is propagated
                        to_search.append(child)
                    elif len(child.halfspaces) > self.maxhsnode:            # If the child is a leaf, the split condition is checked and the search propagated if needed
                        self.splitnode(child)
                        to_search.append(child)

    def getleaves(self):
        """
        Retrieves all leaves of the Qtree. Leaves are node without any children. Non-recursive.
        """

        leaves = []
        to_search = [self.root]

        while len(to_search) > 0:
            current = to_search.pop()

            if current.norm:
                if current.isleaf():
                    leaves.append(current)
                else:
                    to_search += current.children

        return leaves


class QNode:
    """
    Class representing a node of the Quad tree.
    """

    def __init__(self, parent, mbr):
        self.mbr = mbr          # Minimum Bounding Region -> contain the vertices of the quadrant this node covers
        self.norm = True        # If False, the qudrant of this node lies isn't normalized (lies over q1 + q2 + ... + qd = 1) 
        self.order = None       # The number of halfspaces that cover this node
        self.parent = parent
        self.children = []
        self.covered = []
        self.halfspaces = []

    def isroot(self):
        return self.parent is None

    def isleaf(self):
        return len(self.children) == 0

    def getorder(self):
        """
        Computes the order of this node by traversing back the tree. Non-recursive.
        """

        self.order = len(self.covered)
        ref = self.parent

        while not ref.isroot():
            self.order += len(ref.covered)
            ref = ref.parent

        return self.order

    def getcovered(self):
        """
        Retrieves the covering halfspaces by traversing back the tree. Non-recursive.
        """

        covered = self.covered.copy()
        ref = self.parent

        while not ref.isroot():
            covered += ref.covered
            ref = ref.parent

        return covered

    def inserthalfspaces(self, masks, halfspaces):
        incr = (self.mbr[:, 1] - self.mbr[:, 0]) / 2
        half = (self.mbr[:, 0] + self.mbr[:, 1]) / 2
        pts_mask, nds_mask = masks

        pts = incr * pts_mask + half

        coeff = np.array([hs.coeff for hs in halfspaces])
        known = np.array([hs.known for hs in halfspaces])
        pos = np.where(pts.dot(coeff.T) < known, Position.IN, Position.OUT)

        for hs in range(pos.shape[1]):
            rel = np.where(pos[:, hs] != pos[0, hs])

            cross = np.where(np.sum(nds_mask[rel], axis=0) > 0)
            for c in cross[0]:
                self.children[c].halfspaces.append(halfspaces[hs])

            if pos[0, hs] == Position.IN:
                not_cross = np.where(np.sum(nds_mask[rel], axis=0) == 0)
                for nc in not_cross[0]:
                    self.children[nc].covered.append(halfspaces[hs])


def genmasks(dims):
    incr = np.full(dims, 0.5)
    pts = np.full((1, dims), 0.5)

    for d in range(dims):
        lower, higher = np.copy(pts), np.copy(pts)
        lower[:, d] -= incr[d]
        higher[:, d] += incr[d]

        pts = np.vstack((pts, lower, higher))
    pts_mask = (pts - incr) / incr

    mbr = np.empty((2 ** dims, dims, 2))
    for quad in range(2 ** dims):
        qbin = np.array(list(np.binary_repr(quad, width=dims)))

        child_mindim = np.where(qbin == '0', 0, 0.5)
        child_maxdim = np.where(qbin == '1', 1, 0.5)

        mbr[quad] = np.column_stack((child_mindim, child_maxdim))

    nds_mask = np.zeros((pts.shape[0], 2 ** dims), dtype=int)
    for p in range(pts.shape[0]):
        for n in range(2 ** dims):
            if np.all((pts[p] == mbr[n, :, 0]) + (pts[p] == mbr[n, :, 1])):
                nds_mask[p, n] = 1

    return pts_mask, nds_mask
