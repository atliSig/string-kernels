import numpy as np
import pylab

feature_lengths = [3,4,5,6,7,8,10,12,14]

ngram_crude_f1 = [0.94554,0.95722,0.95337,0.96534,0.95346,0.93217,0.93405,0.90452,0.88380]
ngram_crude_p = [0.94303,0.94453,0.93904,0.95330,0.94699,0.91840,0.92733,0.88707,0.89579]
ngram_crude_re = [0.94908,0.97126,0.96997,0.97836,0.96160,0.94933,0.94188,0.92614,0.87454]

ngram_corn_f1 = [0.90720,0.92952,0.92189,0.94116,0.92174,0.88954,0.89357,0.84504,0.81129]
ngram_corn_p = [0.91451,0.95182,0.94997,0.96359,0.93759,0.92040,0.90609,0.88293,0.80124]
ngram_corn_re = [0.90311,0.91073,0.90027,0.92141,0.91053,0.86724,0.88381,0.81833,0.82809]

ngram_earn_f1 = [0.90637,0.90647,0.89605,0.91625,0.91242,0.91221,0.88302,0.83103,0.77092]
ngram_earn_p = [0.95793,0.97972,0.96777,0.97339,0.97566,0.95035,0.92052,0.86535,0.77528]
ngram_earn_re = [0.86352,0.84504,0.83732,0.86944,0.85794,0.88181,0.85345,0.80728,0.77085]

ngram_ac_f1 = [0.91408,0.91943,0.91064,0.92599,0.92102,0.91844,0.88925,0.83732,0.77553]
ngram_ac_p = [0.87498,0.86527,0.85894,0.88414,0.87117,0.89093,0.86429,0.81733,0.77698]
ngram_ac_re = [0.95997,0.98215,0.97151,0.97535,0.97785,0.95206,0.92124,0.86680,0.77814]

ssk_crude_f1 = []
ssk_crude_p = []
ssk_crude_re = []

ssk_corn_f1 = []
ssk_corn_p = []
ssk_corn_re = []

ssk_earn_f1 = []
ssk_earn_p = []
ssk_earn_re = []

ssk_ac_f1 = []
ssk_ac_p = []
ssk_ac_re = []





pylab.rc('grid', linestyle='dashed', color='gray')
pylab.plot(feature_lengths,crude_f1, '-o', label='crude')
pylab.plot(feature_lengths,corn_f1, '-o', color='r', label='corn')
pylab.plot(feature_lengths,earn_f1, '-o', color='g', label='earn')
pylab.plot(feature_lengths,ac_f1, '-o', color='g', label='acquisition')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('Precision')
pylab.title('F1 score of different categories')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()

pylab.rc('grid', linestyle='dashed', color='gray')
pylab.plot(feature_lengths,crude_p, '-o', label='crude')
pylab.plot(feature_lengths,corn_p, '-o', color='r', label='corn')
pylab.plot(feature_lengths,earn_p, '-o', color='g', label='earn')
pylab.plot(feature_lengths,ac_p, '-o', color='g', label='acquisition')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('Precision')
pylab.title('Precision of different categories')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()

pylab.rc('grid', linestyle='dashed', color='gray')
pylab.plot(feature_lengths,crude_re, '-o', label='crude')
pylab.plot(feature_lengths,corn_re, '-o', color='r', label='corn')
pylab.plot(feature_lengths,earn_re, '-o', color='g', label='earn')
pylab.plot(feature_lengths,ac_re, '-o', color='g', label='acquisition')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('Recall')
pylab.title('Recall of different categories')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()