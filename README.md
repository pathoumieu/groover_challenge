## Description

This repo contains notebooks for building a recommender system for influencers and artists/tracks on Groover dataset.

* `01_exploration_and_preprocessing.ipynb` provides quick exploration and preprocessing of categorical data

* `02_first_models.ipynb` builds benchmark with random, naive and LGBM models, and first Neural Networks using Keras

* `03_final_models.ipynb` builds Neural Networks models incorporating textual data from `track_info`


## How to use it

1. install `requirements.txt` in a virtualenv: `pip install -r requirements.txt`
2. create `./data/raw/` and `./data/preprocessed/` directories
3. put raw data in `./data/raw/`
4. install `graphviz`: `sudo apt-get install graphviz`
5. run `01_exploration_and_preprocessing.ipynb` for preprocessing (preprocessed data is created and put in `./data/preprocessed/` directory)
6. run `02_first_models.ipynb` to train models and benchmark
7. download and unzip `cc.fr.300.vec` french pre-trained word embeddings from [this link](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz) and put it in `./`
8. run `03_final_models.ipynb` to train final models
