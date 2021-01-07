# Using Statistical Methods to Build the Optimal Disney Day: An Analysis of Ride Wait Times at Walt Disney World
Repository for files used in development of IW07 project. 
Disclaimer: all files originally used as Python Notebooks and have been adjusted as the project progress and I started working with different data. The file would not run as one whole.

## clean.py
File that contains the code for reading in inital ride wait time data sets. Has many helper functions for cleaning. Contains the functions for merging wait time data for an entire park and filling in missing values. Final computed data tables saved as .csv file.

## kmeans_cluster.py
File that contains the code for splitting the metadata into training and validation sets. Creates 35 K-NN clusters based on training data. Final training and validation values saved as .csv files. 

## predict.py
Imports the computed .csv files from clean.py and kmeans_cluster.ipynb. Contains the algorithms for computing actual shortest paths and predicted shortest paths for a given day and park. Contains much of the evaluation code as well. 
