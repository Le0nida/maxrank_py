import numpy as np

from geom import Point
from query import findknn


class RTree:
    def __init__(self, data, maxpntnode=10):
        self.dims = data.shape[1]
        self.maxpntnode = maxpntnode
        self.root = self.buildtree(data)

    def buildtree(self, data):
        nodes = []

        while len(data) > 0:
            npdata = data.to_numpy()
            argknn = findknn(self.maxpntnode - 1, npdata, npdata[0])
            argknn = np.append(argknn, [0])

            knn = []
            for n in list(argknn):
                knn += [Point(npdata[n], _id=data.index[n])]

            # Compute new mbr
            coords = npdata[argknn]
            child_mindim = np.amin(coords, axis=0)
            child_maxdim = np.amax(coords, axis=0)

            nodes += [RNode(np.column_stack((child_mindim, child_maxdim)), knn)]
            data.drop(data.index[argknn], inplace=True)

        upper_nodes = nodes
        while len(upper_nodes) > 1:
            nodes = upper_nodes
            upper_nodes = []

            while len(nodes) > 0:
                npnodes = np.array([n.mbr for n in nodes])
                argknn = findknn(self.maxpntnode - 1, npnodes[:, :, 0], npnodes[0, :, 0])
                argknn = np.append(argknn, [0])

                coords = np.transpose(npnodes[argknn], (0, 2, 1))
                coords = coords.reshape(-1, coords.shape[-1])
                child_mindim = np.amin(coords, axis=0)
                child_maxdim = np.amax(coords, axis=0)

                knn = [nodes[i] for i in argknn]
                node = RNode(np.column_stack((child_mindim, child_maxdim)), knn)
                for n in knn:
                    n.parent = node
                    nodes.remove(n)

                upper_nodes += [node]

        return upper_nodes[0]


class RNode:
    def __init__(self, mbr, children):
        self.mbr = mbr
        self.parent = []
        self.children = children
