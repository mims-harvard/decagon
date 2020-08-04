# Decagon: Representation Learning on Multimodal Graphs

#### Author: [Marinka Zitnik](http://stanford.edu/~marinka) (marinka@cs.stanford.edu)

#### [Project website](http://snap.stanford.edu/decagon)

## Overview

This repository contains code necessary to run the Decagon algorithm. Code in original repo contains mistakes, so we want to make it runnable on article data.

### Requirements

Code required all packeges from file requirements.txt (latest version preferable) and TensorFlow 2.

### Running the code

The setup for the polypharmacy problem on a synthetic dataset is outlined in `main.py`. It uses a small synthetic 
network example with five edge types. Run the code as following:

    $ python main.py
    
The full polypharmacy dataset (described in the paper) is available on the 
[project website](http://snap.stanford.edu/decagon). To run the code on the full dataset first download all data files
from the [project website](http://snap.stanford.edu/decagon). The polypharmacy dataset is already preprocessed and ready to use. 
After cloning the project, replace the synthetic example in `main.py` with the polypharmacy dataset and run the model.  
