# import pyltr
#
# with open('../data/train.txt') as trainfile, \
#         open('../data/vali.txt') as valifile, \
#         open('../data/test.txt') as evalfile:
#     TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
#     VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
#     EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)
#
# metric = pyltr.metrics.AP()
#
# monitor = pyltr.models.monitors.ValidationMonitor(
#     VX, Vy, Vqids, metric=metric, stop_after=250)
#
# model = pyltr.models.LambdaMART(
#     metric=metric,
#     n_estimators=1500,
#     learning_rate=0.05,
#     max_features=0.5,
#     query_subsample=0.5,
#     max_leaf_nodes=10,
#     min_samples_leaf=64,
#     verbose=1,
# )
#
# model.fit(TX, Ty, Tqids, monitor=monitor)
#
# Epred = model.predict(EX)
# print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
# print('Our model:', metric.calc_mean(Eqids, Ey, Epred))
#
##############################################################

from common import *
from sklearn.model_selection import train_test_split
import features as ft
import fetching as fc

from learning2rank.rank import ListNet

Model = ListNet.ListNet()

joined = pd.read_csv('../data/ftrs.csv.gz')

sims = fc.load_sims('../data/sims.json')
keys_train, keys_test = train_test_split(list(sims.keys()), test_size=0.2, random_state=0)

train = joined[joined['q'].isin(keys_train)]
test = joined[joined['q'].isin(keys_test)]

X_train = train.drop(['q', 'd', 'rank'], axis=1).values
y_train = train['rank'].values

X_test = test.drop(['q', 'd', 'rank'], axis=1).values
y_test = test['rank'].values

Model.fit(X_train, y_train, batchsize=100, n_epoch=200, n_units1=512, n_units2=128, tv_ratio=0.8,
          optimizerAlgorithm="Adam", savefigName="result.pdf", savemodelName="ListNet.model")

print(1)

y_pred = Model.predict(X_test)

import chainer

chainer.using_config