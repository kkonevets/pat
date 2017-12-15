from common import *

'../data/train.txt'

java -jar RankLib.jar -train train.txt -validate vali.txt -ranker 6 -metric2t MAP -save rl_lmart.txt -norm zscore -tts 0.8 -metric2T MAP

java -jar RankLib.jar -train train.txt -validate vali.txt -ranker 7 -tts 0.8 -metric2T MAP -save ln_lmart.txt -norm zscore