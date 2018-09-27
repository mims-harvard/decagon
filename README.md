# Decagon: Representation Learning on Multimodal Graphs

#### Author: [Marinka Zitnik](http://stanford.edu/~marinka) (marinka@cs.stanford.edu)

#### [Project website](http://snap.stanford.edu/decagon)

## Overview

This repository contains code necessary to run the Decagon algorithm. Decagon is a method for learning node 
embeddings in multimodal graphs, and is especially useful for link prediction in highly multi-relational settings. See 
our [paper](https://doi.org/10.1093/bioinformatics/bty294) for details on the algorithm.
  
## Usage: Polypharmacy

Decagon is used to address a burning question in pharmacology, which is that of predicting 
[safety of drug combinations](http://stanford.edu/~marinka/slides/decagon-ismb18.pdf). 

We construct a multimodal graph of protein-protein interactions, drug-protein target interactions, and 
polypharmacy side effects, which are represented as drug-drug interactions, where each side effect is an edge of a 
different type. 

<p align="center">
<img src="https://github.com/marinkaz/decagon/blob/master/images/polypharmacy-graph.png" width="600" align="center">
</p>

Decagon uses graph convolutions to embed the multimodal graph in a compact vector space and then uses
the learned embeddings to predict side effects of drug combinations. 
  
<p align="center">
<img src="https://github.com/marinkaz/decagon/blob/master/images/decagon-architecture-1.png" width="800" align="center">
</p>

### Running the code

The setup for the polypharmacy problem on a synthetic dataset is outlined in `main.py`. It uses a small synthetic 
network example with five edge types. Run the code as following:

    $ python main.py
    
The full polypharmacy dataset (described in the paper) is available on the 
[project website](http://snap.stanford.edu/decagon). To run the code on the full dataset first download all data files
from the [project website](http://snap.stanford.edu/decagon). The polypharmacy dataset is already preprocessed and ready to use. 
After cloning the project, replace the synthetic example in `main.py` with the polypharmacy dataset and run the model.  

## Citing

If you find *Decagon* useful for your research, please consider citing [this paper](https://academic.oup.com/bioinformatics/article/34/13/i457/5045770):

    @article{Zitnik2018,
      title     = {Modeling polypharmacy side effects with graph convolutional networks.},
      author    = {Zitnik, Marinka and Agrawal, Monica and Leskovec, Jure},
      journal   = {Bioinformatics},
      volume    = {34},
      number    = {13},
      pages     = {457–466},
      year      = {2018}
    }

## Miscellaneous

Please send any questions you might have about the code and/or the 
algorithm to <marinka@cs.stanford.edu>.

This code implements several different edge decoders (innerproduct, distmult, 
bilinear, dedicom) and loss functions (hinge loss, cross entropy). Many deep variants are possible and what works 
best might depend on a concrete use case.  

## Requirements

Decagon is tested to work under Python 2 and Python 3. 

Recent versions of Tensorflow, sklearn, networkx, numpy, and scipy are required. All the required packages can be installed using the following command:

    $ pip install -r requirements.txt

## License

Decagon is licensed under the MIT License.
