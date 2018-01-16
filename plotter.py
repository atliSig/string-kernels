import numpy as np
import pylab

feature_lengths = [3,4,5,6,7,8,10,12,14]
earn_p = [0.96029,0.96240,0.96130,0.96285,0.91618,0.90249,0.83110,0.65139,0.59488]
acq_p = [0.92266,0.92789,0.92402,0.91067,0.91342,0.91679,0.74198,0.50192,0.43406]
crude_p = [0.97619, 0.98864, 0.96667, 0.95652, 0.93590, 0.92105, 0.91860, 0.89333, 0.74074 ]
corn_p = [0.75000,0.80882,0.80303,0.81250,0.65385,0.62500,0.70000,0.59259,0.41176]

earn_f1  = [0.95156,0.95464,0.95257,0.94769,0.92794,0.92278,0.81887,0.63055,0.57585]
acq_f1   = [0.93407,0.93810,0.93544,0.92988,0.89660,0.88649,0.75665,0.52267,0.45200]
crude_f1 = [0.89130,0.92553, 0.91579, 0.91667, 0.82022, 0.79545, 0.84946, 0.76571, 0.51948 ]
corn_f1  = [0.84375, 0.88710, 0.86885, 0.86667, 0.76119, 0.73529, 0.77778, 0.70073, 0.53165 ]

earn_r = [0.94300,0.94700,0.94400,0.93300,0.94000,0.94400,0.80700,0.61100,0.55800]
acq_r = [0.94576,0.94854,0.94715,0.94993,0.88039,0.85814,0.77191,0.54520,0.47149]
crude_r = [0.82000,0.87000,0.87000,0.88000,0.73000,0.70000,0.79000,0.67000,0.40000]
corn_r = [0.96429,0.98214,0.94643,0.92857,0.91071,0.89286,0.87500,0.85714,0.75000] 

pylab.rc('grid', linestyle='dashed', color='gray')
pylab.plot(feature_lengths,earn_p, '-o', label='earn')
pylab.plot(feature_lengths,acq_p, '-o', color='r', label='acquisition')
pylab.plot(feature_lengths,crude_p, '-o', color='g', label='crude')
pylab.plot(feature_lengths,corn_p, '-o', color='y', label='corn')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('Precision of category')
pylab.title('Precision of categories')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()

pylab.plot(feature_lengths,earn_f1, '-o', label='earn')
pylab.plot(feature_lengths,acq_f1, '-o', color='r', label='acquisition')
pylab.plot(feature_lengths,crude_f1, '-o', color='g', label='crude')
pylab.plot(feature_lengths,corn_f1, '-o', color='y', label='corn')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('F1 score of category')
pylab.title('F1 score of categories')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()

pylab.plot(feature_lengths,earn_r, '-o', label='earn')
pylab.plot(feature_lengths,acq_r, '-o', color='r', label='acquisition')
pylab.plot(feature_lengths,crude_r, '-o', color='g', label='crude')
pylab.plot(feature_lengths,corn_r, '-o', color='y', label='corn')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('Recall of category')
pylab.title('Recall of categories')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()
