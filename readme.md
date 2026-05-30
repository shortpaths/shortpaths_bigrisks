# Before you begin

There are a few prerequiesites for running the scripts, the first being you will need a NVIDIA GPU for using the CUDA based libraries in the pipeline. The second is that you will need to use Python version 3.11 or later to ensure compatibility with some of the libraries.

There is a requirements.txt file included in the repository that lists all the necessary libraries. There is also a specific requirements_freeze.txt that lists all the libraries and their dependencies and specific versions, these are what CLONEAGE has been developed and tested with.

# VP-Tree and IDF

In cloneage.py, these stages have been disabled by being commented out so that the edges from the full dataset are used. These stages can be commented in and executed but due to their dynamic nature, will give a slightly different results. They remain so they can be tested fully but the edges will be rewritten and IDF recalculated. 

# cloneage_eval.py

This file contains the evaluation for CLONEAGE, conducted as part of Short Paths, Big Risks. When first cloned it only has the last experiment uncommented, simply uncomment whichever experiments are required to be run. 

# CLOENAGE Description
This module implements a multi-stage pipeline to detect structural and evolutionary relationships between browser extensions combining Fuzzy Hashing (TLSH), Information Theory (IDF), Network Theory and Community Detection.

## The Pipeline Architecture

CLONEAGE is divided into five distinct stages:

1. Identity-based Pruning and Pre-processing.
2. Fuzzy Library Clustering. 
3. Cluster-level Structural Weighting.
4. Extension Representation and Similarity
5. Extension Clustering. 

The framework outputs Gephi compatible csv files that can be easily imported, allowing the resultant extension similarity network graph to be visualised and expolred in Gephi.
