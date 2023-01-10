import sys
import numpy as np
import pandas as pd

from geom import *
from maxrank import aa_hd, ba_hd, aa_2d


"""
Main
Run the MaxRank computation by calling : "python main.py path/to/datafile path/to/queryfile method" where:
    datafile: A CSV file containing all data points, correlated with an id. Data should be normalized beforehand, see example folder for reference.
    queryfile: A CSV file contaning the IDs of the points to compute the MaxRank of.
    method: "AA" or "BA". The first method is faster and should always be preferred. See MaxRank paper for reference. NOTE: if DIM = 2, BA isn't even an option.
Example run: "python main.py examples\Test3D50\data_42.csv examples\Test3D50\data_42.csv AA"

The output of the computation consists in two CSV files, "maxrank.csv" and "cells.csv" that will be located in the project root folder.
"maxrank.csv" contains the computed MaxRank of each point listed in the queryfile
"cells.csv" contains the mincells' intervals (DIM = 2) or an example query (DIM > 2)
"""



if __name__ == "__main__":
    datafile = sys.argv[1]
    queryfile = sys.argv[2]
    method = sys.argv[3]

    # Data and query's CSV files are loaded
    data_df = pd.read_csv(datafile, index_col=0)
    query = pd.read_csv(queryfile, index_col=0).index
    print("Loaded {} records from {}\n".format(data_df.shape[0], datafile))

    # Each data point is embedded in a Point object
    data = []
    for i in range(data_df.shape[0]):
        record = data_df.iloc[i]

        data.append(Point(record.to_numpy(), _id=record.name))


    # The main MaxRank routine is called according to the dimensionality and method chosen
    # Each query is handled separately and then the results are glued together
    res = []
    cells = []

    if data_df.shape[1] > 2:
        for q in query:
            print("#  Processing data point {}  #".format(q))
            idx = np.where(data_df.index == q)[0][0]

            print("#  {}  #".format(data[idx].coord))

            if method == 'BA':
                maxrank, mincells = ba_hd(data, data[idx])
            else:
                maxrank, mincells = aa_hd(data, data[idx])
            print("#  MaxRank: {}  NOfMincells: {}  #\n".format(maxrank, len(mincells)))

            res.append([q, maxrank])
            cells.append([q, list(mincells[0].feasible_pnt.coord) + [1 - sum(mincells[0].feasible_pnt.coord)]])
    else:
        for q in query:
            print("#  Processing data point {}  #".format(q))
            idx = np.where(data_df.index == q)[0][0]

            print("#  {}  #".format(data[idx].coord))

            maxrank, mincells = aa_2d(data, data[idx])
            print("#  MaxRank: {}  NOfMincells: {}  #\n".format(maxrank, len(mincells)))

            res.append([q, maxrank])
            cells.append([q, [list(cell.range) for cell in mincells]])

    # The results' CSVs are created
    cells = pd.DataFrame(cells, columns=['id', 'query_found'])
    cells.set_index('id', inplace=True)
    cells.to_csv("./cells.csv")

    res = pd.DataFrame(res, columns=['id', 'maxrank'])
    res.set_index('id', inplace=True)
    res.to_csv("./maxrank.csv")
    print(res)
