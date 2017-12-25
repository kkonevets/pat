Learning to Rank
======

An easy implementation of algorithms of learning to rank. Pairwise (RankNet) and ListWise (ListNet) approach. There implemented also a simple regression of the score with neural network. [Contribution Welcome!]

## Requirements
- python 2.7
- [tqdm](https://github.com/noamraph/tqdm)
- [matplotlib v1.5.1](http://matplotlib.org/)
- [numpy v1.10.1](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [chainer v1.5.1 +](http://chainer.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- and some basic packages.


## RankNet
### Pairwise comparison of rank

The original paper was written by Chris Burges et al., "Learning to Rank using Gradient Descent." (available at http://research.microsoft.com/en-us/um/people/cburges/papers/ICML_ranking.pdf)

### Usage

Import and initialize

```
from learning2rank.rank import RankNet
Model = RankNet.RankNet()
```

Fitting (automatically do training and validation)

```
Model.fit(X, y)
```

Here, `X` is numpy array with the shape of (num_samples, num_features) and `y` is numpy array with the shape of (num_samples, ). `y` is the score which you would like to rank based on (e.g., Sales of the products, page view, etc).

Possible options and defaults:

```
batchsize=100, n_iter=5000, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam", savefigName="result.pdf", savemodelName="RankNet.model"
```

```n_units1``` and ```n_units2=128``` are the number of nodes in hidden layer 1 and 2 in the neural net.

```tv_ratio``` is the ratio of the data amounts between training and validation. 

Predict

```
Model.predict(X)
```

## ListNet

### Listwise comparison of rank

The original paper was written by Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, Hang Li "Learning to Rank: From Pairwise Approach to Listwise Approach." (Available at http://research.microsoft.com/en-us/people/tyliu/listnet.pdf)

NOTICE:
    The top-k probability is not written.
    This is listwise approach with neuralnets, 
    comparing two arrays by Jensen-Shannon divergence.


### Usage

Import and initialize

```
from learning2rank.rank import ListNet
Model = ListNet.ListNet()
```

Fitting (automatically do training and validation)

```
Model.fit(X, y)
```

Same as ranknet, `X` is numpy array with the shape of (num_samples, num_features) and `y` is numpy array with the shape of (num_samples, ). `y` is the score which you would like to rank based on (e.g., Sales of the products, page view, etc).

Possible options and defaults:

```
batchsize=100, n_epoch=200, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam", savefigName="result.pdf", savemodelName="ListNet.model"
```

Predict

```
Model.predict(X)
```



## Regression
### Regression the scores with neural network

### Usage

Import and initialize

```
from learning2rank.regression import NN
Model = NN.NN()
```

Fitting (automatically do training and validation)

```
Model.fit(X, y)
```

Possible options and defaults:

```
batchsize=100, n_iter=5000, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam", savefigName="result.pdf", savemodelName="RankNet.model"
```

```n_units1``` and ```n_units2=128``` are the number of nodes in hidden layer 1 and 2 in the neural net.

```tv_ratio``` is the ratio of the data amounts between training and validation. 

Predict

```
Model.predict(X)
```


## Author

If you have any troubles or questions, please contact [shiba24](https://github.com/shiba24).

March, 2016

