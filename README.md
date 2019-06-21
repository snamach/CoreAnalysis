# CoreAnalysis

SalinityGraph.py is a script which fits salinity data from cores to an overall distribution curve. 

It requries installation of numpy, matplotlib, and scipy libraries for python3.

Save elevations in a file titled "[corename]E.txt" and corresponding salinities in a file titled "[corename]S.txt".

The script provides three fit options (avg, wavg, and aggfit) and three regression options (Exp, Cub, and Sig). 

To run, simply enter the following in the terminal:
python3 [regression] [fit] [corename1] [corename2] [corename3] [etc.]

For example, in order to fit a sigmoidal curve to an aggregate dataset from PM1-1, DP1-1, and DP2-1, enter:

python3 Sig agg PM1-1 DP1-1 DP2-1
