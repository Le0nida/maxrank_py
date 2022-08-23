import numpy as np
import pandas as pd

from geom import *
from maxrank import aa_hd, ba_hd

print("\n"*25)

if __name__ == "__main__":
    df = pd.read_csv("examples/Test3D50/data_42.csv", index_col=0)
    print("Loaded {} records from examples/Test3D50/data_42.csv\n".format(df.shape[0]))

    data = []
    for i in range(df.shape[0]):
        record = df.iloc[i]

        data.append(Point(record.name, record.to_numpy()))

    queries = np.empty(shape=(len(data), 2), dtype=int)
    for i in range(len(data)):
        print("#  Processing data point {}  #".format(data[i].id))
        maxrank, mincells = ba_hd(data, data[i])
        print("#  MaxRank: {}  NOfMincells: {}  #\n".format(maxrank, len(mincells)))
        queries[i] = [data[i].id, maxrank]

    res = pd.DataFrame(queries, columns=['id', 'maxrank'])
    res.set_index('id', inplace=True)
    res.to_csv("examples/Test3D50/maxrank.csv")
    print(res)
