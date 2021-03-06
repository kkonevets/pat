from common import *
from sklearn.model_selection import train_test_split
from itertools import *
from collections import Counter
import cProfile, pstats
from operator import itemgetter
from scipy.spatial import distance
from importlib import reload
from functools import partial
import features as ft
import fetching as fc


reload(ft)
reload(fc)

SEED = 0

#   ########################### tfidf model ####################################

dictionary = Dictionary.load('../data/corpus.dict')
corpus = MmCorpus('../data/corpus.mm')
# build_tfidf_index(dictionary, corpus, anew=True)

index = Similarity.load('../data/sim_index/sim')
tfidf = TfidfModel.load('../data/tfidf.model')

#   ####################### prepare data ###############################

corpus_files = glob('../data/documents/*')
corpus_files.sort(key=natural_keys)

all_ids = fc.load_keys('../data/keys.json')
ix_map = {vi: i for i, vi in enumerate(all_ids)}
sims = fc.load_sims('../data/sims.json')
sims = {ix_map[k]: [ix_map[vi] for vi in v] for k,v in sims.items()}
with open('../data/gold_mongo.json', 'r') as f:
    gold = json.load(f)
gold = {ix_map[k]: [ix_map[vi] for vi in v] for k,v in gold.items()}

with open('../data/sampled.json', 'r') as f:
    samples = json.load(f)

# ############################ small test  ##################################

with open('../data/gold_mongo.json', 'r') as f:
    gold = json.load(f)

_gold = {ix_map[k]: [ix_map[vi] for vi in v] for k,v in gold.items()}


tfidf_blob = ft.TfIdfBlob(corpus, tfidf, index, all_ids)
preds = tfidf_blob.predict(_gold.keys(), limit=201)
preds = preds[:, 1:]
preds = {k: [all_ids[ix] for ix in preds[i]] for i,k in enumerate(gold.keys())}
res = evaluate(preds, gold)
"""
acc10 0.286738
acc20 0.347670
acc200 0.573477
"""

# ############################# sample negs ############################

df = pd.read_csv('../data/foundat.csv', header=None, names=['rank'])
# df.plot.hist(bins=1000)
df.describe()
neg_ixs_distr = df['rank'].values

tfidf_blob = ft.TfIdfBlob(corpus, tfidf, index, all_ids)
ns = ft.NegativeSampler(sims, tfidf_blob, neg_ixs_distr,
                 k=1, percentile=90, seed=SEED)
samples = ns.gen_samples('../data/sampled.json', n_chunks=500)

#   ############################# gensim tfidf ##############################

tfidf_blob = ft.TfIdfBlob(corpus, tfidf, index, all_ids)
tfidf_scores = tfidf_blob.extract(samples, '../data/tfidf.csv', n_chunks=150)

# ################################## QDR #####################################

pairs = list(chain.from_iterable([[(anc, i) for i in pos + neg]
                             for anc, pos, neg in samples]))
qdr = ft.QDR(fmodel='../data/qdr_model.gz')
# qdr.train(corpus_files)
qdr.load()

corpus = MmCorpus('../data/corpus.mm')

qdr_ftrs = qdr.extract(pairs, all_ids, corpus, '../data/qdr_ftrs.csv')

# qdr.describe_bm25(qdr_ftrs)

# qdr.bm25_sanity_check(corpus, samples)

# ############################## gen features ##################################

qdr_ftrs = pd.read_csv('../data/qdr.csv')
tfidf_ftrs = pd.read_csv('../data/tfidf.csv')

joined = qdr_ftrs.merge(tfidf_ftrs, on=['q', 'd'])

#   ###################### independent and jaccard #############################

unique = chain.from_iterable([[anc] + pos + neg for anc, pos, neg in samples])
unique = set([all_ids[el] for el in unique])

indep = ft.Independent(corpus_files, unique)
ind_ftrs = indep.extract('../data/independ_ftrs.csv')

ind_ftrs = pd.read_csv('../data/independ_ftrs.csv')


jac = ft.Jaccard(unique, dictionary.token2id, corpus_files)
jac.extract(samples, all_ids, '../data/jaccard.csv')

jaccard = pd.read_csv('../data/jaccard.csv')

assert len(joined) == len(jaccard)

#   ######################## word2vec ####################################

sents = fc.Sentences('../data/documents/')
model = Word2Vec(sents, size=200,
                 min_count=5, window=8, workers=cpu_count)
model.save('../data/w2v_200_5_w8')

model = Word2Vec.load('../data/w2v_200_5_w8')

# for w ,s in model.most_similar('стол', topn=10):
#     print('%s %s' % (w,s))

wv = model.wv
ids = list(chain.from_iterable([[anc] + pos + neg for anc, pos, neg in samples]))
ids = set([all_ids[el] for el in ids])
index2word = wv.index2word
docs_in_ram = ft.push_docs_to_ram(ids, wv.vocab,
                                        corpus_files, is_gensim=True)

ftrs = []
for samp_part in tqdm(chunkify(samples, 50)):
    res = Parallel(n_jobs=cpu_count, backend="multiprocessing") \
        (delayed(ft.distributed_worker)(all_ids, docs_in_ram, index2word, wv, part) for
         part in chunkify(samp_part, cpu_count))
    ftrs += list(chain.from_iterable(res))

cosines = ft.to_dataframe(ftrs)
ft.save(cosines, '../data/cosines.csv')

cosines = pd.read_csv('../data/cosines.csv')

#   ############################### MPK #######################################

all_mpk = fc.MPKFetcher().fetch_all(all_ids, fname='../data/all_mpk.pkl')

with open('../data/all_mpk.pkl', 'rb') as f:
    all_mpk = pickle.load(f)

mpk_ftrs = ft.MPK().extract(all_mpk, samples, all_ids, '../data/mpk.csv')
mpk_ftrs = pd.read_csv('../data/mpk.csv')
mpk_ftrs.drop_duplicates(['q', 'd'], inplace=True)

#   ################################# unite features #############################

qdr_ftrs = pd.read_csv('../data/qdr.csv')
qdr_ftrs.drop_duplicates(['q', 'd'], inplace=True)
tfidf_ftrs = pd.read_csv('../data/tfidf.csv')
tfidf_ftrs.drop_duplicates(['q', 'd'], inplace=True)
joined0 = qdr_ftrs.merge(tfidf_ftrs, on=['q', 'd'])

ind_ftrs = pd.read_csv('../data/independ.csv')
ind_ftrs.drop_duplicates(['q'], inplace=True)
joined = joined0.merge(ind_ftrs, on='q')
cp = ind_ftrs.copy()
cp['d'] = cp['q']
del cp['q']
joined = joined.merge(cp, on='d', suffixes=('_q', '_d'))

assert len(joined0) == len(joined)

jaccard = pd.read_csv('../data/jaccard.csv')
jaccard.drop_duplicates(['q', 'd'], inplace=True)
joined = joined.merge(jaccard, on=['q', 'd'])
joined.drop_duplicates(['q', 'd'], inplace=True)
pprint('%s %s' % (len(joined0), len(joined)))
pprint(joined.isnull().sum())


ranks = []
for anc, pos, neg in samples:
    q_id = all_ids[anc]
    lst = [(q_id, all_ids[p], 1) for p in pos]
    ranks += lst
    lst = [(q_id, all_ids[n], 0) for n in neg]
    ranks += lst

ranks = pd.DataFrame(ranks, columns=['q', 'd', 'rank'])
ranks.drop_duplicates(['q', 'd'], inplace=True)
joined = joined.merge(ranks, on=['q', 'd'])

cosines = pd.read_csv('../data/cosines.csv')
joined = joined.merge(cosines, on=['q', 'd'])
joined.drop_duplicates(['q', 'd'], inplace=True)

mpk_ftrs = pd.read_csv('../data/mpk.csv')
mpk_ftrs.drop_duplicates(['q', 'd'], inplace=True)
joined = joined.merge(mpk_ftrs, on=['q', 'd'])

joined.sort_values(['q', 'rank'], inplace=True, ascending=(True, False))
joined.set_index(['q', 'd', 'rank'], inplace=True)

joined.to_csv('../data/ftrs.csv.gz', compression='gzip')
joined[:10000].to_csv('../data/ftrs_show.csv')

joined = pd.read_csv('../data/ftrs.csv.gz')

#   ######################### LETOR format ####################################

sims = fc.load_sims('../data/sims.json')
keys_tv, keys_test = train_test_split(list(sims.keys()), test_size=0.2, random_state=SEED)
keys_train, keys_val = train_test_split(keys_tv, test_size=0.2, random_state=SEED)

train = joined[joined['q'].isin(keys_train)]
vali = joined[joined['q'].isin(keys_val)]
test = joined[joined['q'].isin(keys_test)]

subtrain = train.drop(['q', 'd', 'rank'], axis=1)
subvali = vali.drop(['q', 'd', 'rank'], axis=1)
subtest = test.drop(['q', 'd', 'rank'], axis=1)

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(subtrain)


def scale(df, scaler):
    subdf = df.drop(['q', 'd', 'rank'], axis=1)
    subdf = scaler.transform(subdf)
    subdf = np.hstack((df[['q', 'd', 'rank']], subdf))
    return pd.DataFrame(subdf, columns=df.columns)


train = scale(train, scaler)
vali = scale(vali, scaler)
test = scale(test, scaler)

ft.save_letor(train, '../data/train.txt')
ft.save_letor(vali, '../data/vali.txt')
ft.save_letor(test, '../data/test.txt')

########################## test 184 ###############################################

with open('../data/gold_mongo.json', 'r') as f:
    gold = json.load(f)
_gold = {ix_map[k]: [ix_map[vi] for vi in v] for k,v in gold.items()}


tfidf_blob = ft.TfIdfBlob(corpus, tfidf, index, all_ids)
cosines = tfidf_blob.get_cosines(_gold.keys())
preds = np.argsort(-cosines)
sorted_cosines = -np.sort(-cosines)
preds = preds[:,1:]
sorted_cosines = sorted_cosines[:,1:]


found_at = []
keys = list(_gold.keys())
for i, iix in enumerate(preds):
    ix = keys[i]
    posvs = _gold[ix]
    ixs = np.isin(iix, posvs)
    found_at += list(np.where(ixs)[0])


pd.Series(found_at).describe().plot.hist(bins=100)


test_ixs = []
keys = list(_gold.keys())
for i, (iix, coss) in enumerate(zip(preds, sorted_cosines)):
    ix = keys[i]
    posvs = _gold[ix]
    ixs = np.isin(iix, posvs)
    mix = max(np.where(ixs)[0])
    up_to = max(13000, mix + 10)
    test_ixs.append(list(zip(iix[:up_to], coss[:up_to])))

test_ixs = [[(int(i), float(cos)) for i,cos in el] for el in test_ixs]
with open('../data/test_ixs.json', 'w') as f:
    json.dump(test_ixs, f)




























































