from common import *

'../data/train.txt'

java -jar RankLib.jar -train train.txt -validate vali.txt -ranker 6 -metric2t MAP -save rl_lmart.txt