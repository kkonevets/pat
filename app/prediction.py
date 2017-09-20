import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import euclidean_distances


def predict(vectors, vectors_norm_squared, names, keys, lim=200):
    test_vecs = []
    for k in keys:
        ix = names[k]
        if isinstance(ix, pd.Series):
            ix = ix[0]
        tvec = vectors[ix]
        test_vecs.append(tvec)

    start_time = time.time()

    # most time consuming point
    dists = euclidean_distances(test_vecs, vectors, 
        Y_norm_squared=vectors_norm_squared)
    sorted_ixs = np.argsort(dists, axis=1)

    print("--- %s seconds ---" % (time.time() - start_time))

    preds = {}
    for k, _ixs in zip(keys, sorted_ixs):
        preds[k] = [n for n in names.index[_ixs[1:lim+1]]]
        
    return preds