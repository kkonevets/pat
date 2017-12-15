from common import *
from sklearn.model_selection import train_test_split
from itertools import *
from collections import Counter
import cProfile, pstats
from importlib import reload
import features as ft
import fetching as fc


reload(ft)
reload(fc)

SEED = 0

#   ########################### tfidf ####################################

dictionary = Dictionary.load('../data/corpus.dict')
corpus = MmCorpus('../data/corpus.mm')
# build_tfidf_index(dictionary, corpus, anew=True)

index = Similarity.load('../data/sim_index/sim')
tfidf = TfidfModel.load('../data/tfidf.model')

#   ####################### fetch ids data ###############################

corpus_files = glob('../data/documents/*')
corpus_files.sort(key=natural_keys)

all_ids = fc.load_keys('../data/keys.json')
ix_map = {vi: i for i, vi in enumerate(all_ids)}
sims = fc.load_sims('../data/sims.json')
sims = {ix_map[k]: [ix_map[vi] for vi in v] for k,v in sims.items()}
with open('../data/gold_mongo.json', 'r') as f:
    gold = json.load(f)
gold = {ix_map[k]: [ix_map[vi] for vi in v] for k,v in gold.items()}

# ############################ small test  ##################################

tfidf_blob = ft.TfIdfBlob(corpus, tfidf, index)
preds = tfidf_blob.predict(gold.keys(), limit=200)
res = evaluate(preds, gold)
"""
acc10 0.286738
acc20 0.347670
acc200 0.573477
"""

# ############################# sample negs ############################

df = pd.read_csv('../data/foundat.csv', header=None, names=['rank'])
# df.plot.hist(bins=100)
df.describe()
neg_ixs_distr = df['rank'].values

fname = '../data/sampled.json'
tfidf_blob = ft.TfIdfBlob(corpus, tfidf, index)
ns = ft.NegativeSampler(sims, tfidf_blob, neg_ixs_distr,
                 k=1, percentile=90, seed=SEED)
ns.gen_samples(fname, n_chunks=500)

with open(fname, 'r') as f:
    samples = json.load(f)

# ################################## QDR #####################################

qdr = ft.QDR('../data/qdr_model.gz')
# qdr.train(corpus_files)
qdr.load()

corpus = MmCorpus('../data/corpus.mm')

tfidf_blob.save_tfidf_features(samples, '../data/tfidf_scores.pkl')
qdr.save_qdr_features(corpus, samples)

with open('../data/qdr_scores.pkl', 'rb') as f:
    qdr_scores = pickle.load(f)
with open('../data/tfidf_scores.pkl', 'rb') as f:
    tfidf_scores = pickle.load(f)

i = 0
pprint([s if type(s) == int else [si['tfidf'] for si in s] for s in qdr_scores[i]])

pprint(qdr_scores[i])
pprint(tfidf_scores[i])

qdr.describe_bm25(qdr_scores)

qdr.bm25_sanity_check(corpus, samples)

# ############################## gen features ##################################

names = ['q', 'rank', 'tfidf_qdr', 'bm25', 
    'lm_jm', 'lm_dirichlet', 'lm_ad', 'tfidf_gs', 'd']
scores_list = (qdr_scores, tfidf_scores)

ft.save_ftrs_to_dataframe(scores_list, names, samples, all_ids)
ftrs_df = pd.read_csv('../data/qdr_gens_ftrs.csv')

#   ###################### independent and jaccard #############################

corpus_files = glob('../data/documents/*')
corpus_files.sort(key=natural_keys)
# all_ids = load_keys('../data/keys.json')

unique = set(ftrs_df['q']).union(ftrs_df['d'])

ftrs_independent = pd.read_csv('../data/independ_ftrs.csv')


ids = set(ftrs_df['q']).union(ftrs_df['d'])
jac = ft.Jaccard(ids, dictionary.token2id, corpus_files)


jaccard = pd.read_csv('../data/jaccard.csv')

assert len(ftrs_df) == len(jaccard)

#   ######################## word2vec ####################################

sents = fc.Sentences('../data/documents/')
model = Word2Vec(sents, size=200,
    min_count=5, window=8, workers=cpu_count)
model.save('../data/w2v_200_5_w8')

all_ids = fc.load_keys('../data/keys.json')
model = Word2Vec.load('../data/w2v_200_5_w8')
ftrs_df = pd.read_csv('../data/qdr_gens_ftrs.csv')
with open('../data/sampled.json', 'r') as f:
    samples = json.load(f)

# for w ,s in model.most_similar('стол', topn=10):
#     print('%s %s' % (w,s))

docs_in_ram = ft.push_docs_to_ram(model.wv.vocab, is_gensim=True)

disted = ft.Distribured(model, docs_in_ram, all_ids)
cosines = disted.gen_ftrs(samples, '../data/cosines.csv', n_chunks=50)

cosines = pd.read_csv('../data/cosines.csv')

#   ############################### MPK #######################################

all_ids = fc.load_keys('../data/keys.json')
with open('../data/sampled.json', 'r') as f:
    samples = json.load(f)

all_mpk = ft.MPKFetcher().fetch_all(samples, all_ids, fname='../data/all_mpk.pkl')

with open('../data/all_mpk.pkl', 'rb') as f:
    all_mpk = pickle.load(f)

mpk_ftrs = pd.read_csv('../data/mpk_ftrs.csv')

#   ################################# unite features #############################

joined = ftrs_df.merge(ftrs_independent, on='q')
cp = ftrs_independent.copy()
cp['d'] = cp['q']
del cp['q']
joined = joined.merge(cp, on='d', suffixes=('_q', '_d'))

assert len(ftrs_df) == len(joined)

joined = joined.merge(jaccard, on=['q', 'd'])
joined.drop_duplicates(['q', 'd'], inplace=True)
pprint('%s %s' % (len(ftrs_df), len(joined)))
pprint(joined.isnull().sum())

for _ix, g in tqdm(joined.groupby(['q'])):
    assert len(g['rank'].unique()) > 1
    break

joined = joined.merge(cosines, on=['q', 'd'])

joined = joined.merge(mpk_ftrs, on=['q', 'd'])

joined.sort_values(['q', 'rank'], inplace=True)
joined.set_index(['q', 'd'], inplace=True)

joined.to_csv('../data/ftrs.csv.gz', compression='gzip')
joined[:10000].to_csv('../data/ftrs_show.csv')

joined = pd.read_csv('../data/ftrs.csv.gz')

#   ######################### LETOR format #############################

sims = fc.load_sims('../data/sims.json')
keys_tv, keys_test = train_test_split(list(sims.keys()), test_size=0.2, random_state=SEED)
keys_train, keys_val = train_test_split(keys_tv, test_size=0.2, random_state=SEED)

train = joined[joined['q'].isin(keys_train)]
val = joined[joined['q'].isin(keys_val)]

ft.save_letor(train, '../data/train.txt')
ft.save_letor(val, '../data/vali.txt')

