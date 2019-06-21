# CoreAnalysis

SalinityGraph.py is a script which fits salinity data from cores to an overall distribution curve. 

It requries installation of NumpPy, Matplotlib, and SciPy libraries for python3.

Save elevations in a file titled "[corename]E.txt" and corresponding salinities in a file titled "[corename]S.txt".

The script provides three regression options (Exp, Cub, and Sig) and three fit options (avg, wavg, and agg). 

To run, simply enter the following in the terminal:

python3 SalinityGraph.py [regression] [fit] [corename1] [corename2] [corename3] [etc.]

For example, in order to fit an aggregate dataset from PM1-1, DP1-1, and DP2-1 to a sigmoidal curve, enter:

python3 SalinityGraph.py Sig agg PM1-1 DP1-1 DP2-1
