
from common import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, \
    roc_auc_score, confusion_matrix
from sklearn.utils import shuffle

SEED = 0


ftrs_df = pd.read_csv('../data/qdr_gens_ftrs.csv')

data = ftrs_df.drop(columns=['Unnamed: 0', 'q', 'd', 'rank'])

from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
import seaborn as sns

train_val = ftrs_df['q'].unique()
t_ixs, v_ixs = train_test_split(train_val, test_size=0.2, random_state=SEED)
x_train = data.loc[ftrs_df['q'].isin(t_ixs)]
y_train = ftrs_df.loc[ftrs_df['q'].isin(t_ixs),'rank']
x_val = data.loc[ftrs_df['q'].isin(v_ixs)]
y_val = ftrs_df.loc[ftrs_df['q'].isin(v_ixs),'rank']

x_train, y_train = shuffle(x_train, y_train, random_state=SEED)
x_val, y_val = shuffle(x_val, y_val, random_state=SEED)

scaler = StandardScaler()
print(scaler.fit(x_train))

x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
x_val = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns)

plt.scatter(x_val['bm25'], x_val['tfidf_gs'], c=y_val, alpha=0.1)
plt.show()

from sklearn import linear_model

model = linear_model.LogisticRegression(C=1)
model.fit(x_train, y_train)
probs = model.predict_proba(x_val)[:,0]

ones = probs[np.where(y_val == 1)]
twoes = probs[np.where(y_val == 2)]
ft.evaluate_probs(probs, y_val == 1, threshold=0.7)

sns.distplot(ones, label='sim')
sns.distplot(twoes, label='dissim')
plt.xlabel('prob')
plt.ylabel('freq')
plt.legend(loc="best")
plt.show(block=False)


probs = model.predict_proba(x_train)[:,0]
ones = probs[np.where(y_train == 1)]
twoes = probs[np.where(y_train == 2)]
ft.evaluate_probs(probs, y_train == 1, threshold=0.3)


# WHY NOT OVERFIT ON TRAIN DATA WITH XGBOOST???

import xgboost as xgb

model = xgb.XGBClassifier(
    max_depth=5,
    n_estimators=2000,
    learning_rate=0.3)
model.fit(x_train, y_train)

probs = model.predict_proba(x_train)[:,0]
ones = probs[np.where(y_train == 1)]
twoes = probs[np.where(y_train == 2)]
ft.evaluate_probs(probs, y_train == 1, threshold=0.45)

probs = model.predict_proba(x_val)[:,0]
ones = probs[np.where(y_val == 1)]
twoes = probs[np.where(y_val == 2)]
ft.evaluate_probs(probs, y_val == 1, threshold=0.34)


print(shuffle([1,2,3,4], random_state=1))

roc_auc_score(y_train==1, probs)

y_pred = [1 if p >= 0.35 else 2 for p in probs]
accuracy_score(y_train, y_pred)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_val, y_pred, average='macro')
