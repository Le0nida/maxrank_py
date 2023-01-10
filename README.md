<p align="center">
  <img width="300" src="https://i.imgur.com/OQ1OjG9.png" alt="PoliMi Logo" />
  <br>
  <i><font size="3">
  	Master Thesis Side Project - AA 2021/2022 - Prof. Davide Martinenghi
    </font>
  </i>
</p>
<h1 align="center">
	<strong>
	MaxRank
	</strong>
	<br>
</h1>
<p align="center">
<font size="3">	
		<a href="https://www.politesi.polimi.it/">Thesis</a>		 
		•
		<a href="https://doi.org/10.14778/2824032.2824053">Original Paper</a>   
	</font>
</p>

A Python implementation of the algorithms described by Mouratidis et al. in "Maximum Rank Query" for the calculation of a record's MaxRank.

## Requisites
The project requires the following libraries:
* NumPy
* SciPy
* pandas
* sortedcontainers

The latest releases fetched by pip should be fine.

## Usage
Run the MaxRank computation by calling:
```console
python main.py path\to\datafile.csv path\to\queryfile.csv method
```
* **datafile.csv** is a CSV file containing all data points, correlated with an ID.
* **queryfile.csv** is a CSV file contaning the IDs of the points to compute the MaxRank of.
* The **method** should be "AA" or "BA". The first one is faster and should be preferred. See the original paper for reference. 

Example run:
```console
python main.py examples\Test3D50\data_42.csv examples\Test3D50\data_42.csv AA
```

## Output
The output of the computation consists in two CSV files, that will be located in the project root folder.
* **maxrank.csv** contains the computed MaxRank of each point listed in the queryfile.
* **cells.csv** contains the mincells' intervals (DIM = 2) or an example query (DIM > 2).

## Notes
This Python implemetation may be slow for very large and/or highly dimensional datasets. If the computation gets unfeasibly long, try checking out the old [C++ implementation](https://github.com/MarcoSomaschini/maxrank).

## Author
*Somaschini Marco*