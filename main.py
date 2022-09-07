import sys
import numpy as np
import pandas as pd

from geom import *
from maxrank import aa_hd, ba_hd, aa_2d

print("\n" * 25)

if __name__ == "__main__":
    datafile = sys.argv[1]
    queryfile = sys.argv[2]
    method = sys.argv[3]

    data_df = pd.read_csv(datafile, index_col=0)
    query = pd.read_csv(queryfile, index_col=0).index
    print("Loaded {} records from {}\n".format(data_df.shape[0], datafile))

    data = []
    for i in range(data_df.shape[0]):
        record = data_df.iloc[i]

        data.append(Point(record.to_numpy(), _id=record.name))

    # rt = RTree(df, maxpntnode=5)

    res = []
    cells = []

    if data_df.shape[1] > 2:
        for q in query:
            print("#  Processing data point {}  #".format(q))
            idx = np.where(data_df.index == q)[0][0]

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

            maxrank, mincells = aa_2d(data, data[idx])
            print("#  MaxRank: {}  NOfMincells: {}  #\n".format(maxrank, len(mincells)))

            res.append([q, maxrank])
            cells.append([q, [list(cell.range) for cell in mincells]])

    cells = pd.DataFrame(cells, columns=['id', 'query_found'])
    cells.set_index('id', inplace=True)
    cells.to_csv("./cells.csv")

    res = pd.DataFrame(res, columns=['id', 'maxrank'])
    res.set_index('id', inplace=True)
    res.to_csv("./maxrank.csv")
    print(res)
