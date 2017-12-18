from common import *

'../data/train.txt'

java -jar RankLib.jar -train train.txt -validate vali.txt -ranker 6 -metric2t MAP -save rl_lmart.txt -norm zscore -test test.txt -metric2T MAP

java -jar RankLib.jar -train train.txt -validate vali.txt -ranker 7 -test test.txt -metric2T MAP -save ln_lmart.txt -norm zscore