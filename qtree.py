import numpy as np

from geom import *


class QTree:
    def __init__(self, dims, maxhsnode):
        self.dims = dims
        self.maxhsnode = maxhsnode
        self.root = self.createroot()

    def createroot(self):
        root = QNode(None, np.column_stack(([0.0 for d in range(self.dims)], [1.0 for d in range(self.dims)])))
        self.splitnode(root)

        return root

    def splitnode(self, node):
        mindim = node.mbr[:, 0]
        maxdim = node.mbr[:, 1]

        # The number of quadrants is dependant by the dimensionality
        for quad in range(2 ** self.dims):
            # Convert the quadrant number in binary
            qbin = np.binary_repr(quad, width=self.dims)

            # Compute new mbr
            child_mindim = [mindim[d] if qbin[d] == '0' else (mindim[d] + maxdim[d]) / 2 for d in range(self.dims)]
            child_maxdim = [maxdim[d] if qbin[d] == '1' else (mindim[d] + maxdim[d]) / 2 for d in range(self.dims)]

            child = QNode(node, np.column_stack((child_mindim, child_maxdim)))

            # Insert parent halfspaces in child
            for hs in node.halfspaces:
                self.inserthalfspace(child, hs)  # Recursive

            node.children.append(child)
        node.halfspaces = []

    def inserthalfspace(self, node, halfspace):
        mindim = node.mbr[:, 0]
        maxdim = node.mbr[:, 1]

        n_above = 0
        n_below = 0
        n_on = 0
        # Get corner points of the quadrant
        for corner in range(2 ** self.dims):
            cbin = np.binary_repr(corner, width=self.dims)
            corner_pnt = Point(None, np.array([mindim[d] if cbin[d] == '0' else maxdim[d] for d in range(self.dims)]))

            # Find the postion of the corner w.r.t the halfspace
            rec_position = find_pointhalfspace_position(corner_pnt, halfspace)

            if rec_position is Position.ABOVE:
                n_above += 1
            elif rec_position is Position.BELOW:
                n_below += 1
            else:
                n_on += 1

        # Deduce the position of the quadrant
        if n_above + n_on == 2 ** self.dims:  # If the halfspace is above it's a covering halfspace
            node.covered.append(halfspace)
        elif not n_below + n_on == 2 ** self.dims:  # If the halfspace is not above nor below it crosses the quadrant
            if node.isleaf():
                node.halfspaces.append(halfspace)
            else:
                for child in node.children:
                    self.inserthalfspace(child, halfspace)  # Recursive

        # Futher split the node if necessary
        if len(node.halfspaces) > self.maxhsnode:
            self.splitnode(node)

    def getleaves(self):
        leaves = []
        to_search = [self.root]

        while len(to_search) > 0:
            current = to_search.pop()

            if current.isleaf():
                leaves.append(current)
            else:
                to_search = to_search + current.children

        return leaves


class QNode:
    def __init__(self, parent, mbr):
        self.mbr = mbr
        self.order = None
        self.parent = parent
        self.children = []
        self.covered = []
        self.halfspaces = []

    def isroot(self):
        return self.parent is None

    def isleaf(self):
        return len(self.children) == 0

    def getorder(self):
        order = len(self.covered)
        ref = self.parent

        while not ref.isroot():
            order += len(ref.covered)
            ref = ref.parent

        self.order = order
        return order
