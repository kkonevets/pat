import pyltr

with open('../data/train.txt') as trainfile, \
        open('../data/vali.txt') as valifile, \
        open('../data/test.txt') as evalfile:
    TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
    VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
    EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

metric = pyltr.metrics.AP()

monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1500,
    learning_rate=0.05,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(TX, Ty, Tqids, monitor=monitor)

Epred = model.predict(EX)
print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
print('Our model:', metric.calc_mean(Eqids, Ey, Epred))


##############################################################


from learning2rank.rank import ListNet
Model = ListNet.ListNet()
