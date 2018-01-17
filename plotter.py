import numpy as np
import pylab

feature_lengths = [3,4,5,6,7,8]

earn_p_1 = [0.86712, 0.97552, 0.96407, 0.88474, 0.88549, 0.58117]
earn_p_2 = [0.80572,0.85914,0.89599,0.96305,0.82099,0.59903]
earn_p_3 = [0.75907,0.82927,0.84079,0.75470,0.67458,0.55833]

acq_p_1 = [0.62543,0.60718,0.60345,0.64970,0.58005,0.51558]
acq_p_2 = [0.84776,0.74244,0.72578,0.63483,0.58849,0.54061]
acq_p_3 = [0.84411,0.77567,0.70827,0.63501,0.56535,0.53209]

earn_f1_1 = [0.32556,0.51546,0.52239,0.59773,0.44900,0.25021]
earn_f1_2 = [0.82512,0.78161,0.74985,0.63026,0.46767,0.37740]
earn_f1_3 = [0.80017,0.79023,0.73884,0.60739,0.46563,0.44068]

acq_f1_1 = [0.63993,0.75102,0.74722,0.76643,0.72153,0.64130]
acq_f1_2 = [0.81296,0.80061,0.81116,0.76993,0.71548,0.65158]
acq_f1_3 = [0.76782,0.80545,0.77765,0.71429,0.67283,0.60868]

earn_r_1 = [0.35440,0.36052,0.36351,0.45895,0.30763,0.17433]
earn_r_2 = [0.85231,0.71868,0.64998,0.47103,0.34549,0.27567]
earn_r_3 = [0.85811,0.75773,0.66213,0.52061,0.35919,0.36572]

acq_r_1 = [0.81090,0.99016,0.98323,0.93884,0.95628,0.85796]
acq_r_2 = [0.78946,0.87018,0.92361,0.97927,0.92065,0.81993]
acq_r_3 = [0.71810,0.84020,0.86507,0.82580,0.83288,0.71229]


pylab.rc('grid', linestyle='dashed', color='gray')
pylab.plot(feature_lengths,earn_p_1, '-o', label='$\lambda=0.1$')
pylab.plot(feature_lengths,earn_p_2, '-o', color='r', label='$\lambda=0.5$')
pylab.plot(feature_lengths,earn_p_3, '-o', color='g', label='$\lambda=0.9$')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('Precision')
pylab.title('Comparing precision with varying $\lambda$ values')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()

pylab.rc('grid', linestyle='dashed', color='gray')
pylab.plot(feature_lengths,acq_p_1, '-o', label='$\lambda=0.1$')
pylab.plot(feature_lengths,acq_p_2, '-o', color='r', label='$\lambda=0.5$')
pylab.plot(feature_lengths,acq_p_3, '-o', color='g', label='$\lambda=0.9$')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('Precision')
pylab.title('Comparing precision with varying $\lambda$ values')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()

pylab.rc('grid', linestyle='dashed', color='gray')
pylab.plot(feature_lengths,earn_f1_1, '-o', label='$\lambda=0.1$')
pylab.plot(feature_lengths,earn_f1_2, '-o', color='r', label='$\lambda=0.5$')
pylab.plot(feature_lengths,earn_f1_3, '-o', color='g', label='$\lambda=0.9$')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('Precision')
pylab.title('Comparing F1 score with varying $\lambda$ values')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()

pylab.rc('grid', linestyle='dashed', color='gray')
pylab.plot(feature_lengths,acq_f1_1, '-o', label='$\lambda=0.1$')
pylab.plot(feature_lengths,acq_f1_2, '-o', color='r', label='$\lambda=0.5$')
pylab.plot(feature_lengths,acq_f1_3, '-o', color='g', label='$\lambda=0.9$')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('Precision')
pylab.title('Comparing F1 score with varying $\lambda$ values')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()

pylab.rc('grid', linestyle='dashed', color='gray')
pylab.plot(feature_lengths,earn_r_1, '-o', label='$\lambda=0.1$')
pylab.plot(feature_lengths,earn_r_2, '-o', color='r', label='$\lambda=0.5$')
pylab.plot(feature_lengths,earn_r_3, '-o', color='g', label='$\lambda=0.9$')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('Precision')
pylab.title('Comparing recall with varying $\lambda$ values')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()

pylab.rc('grid', linestyle='dashed', color='gray')
pylab.plot(feature_lengths,acq_r_1, '-o', label='$\lambda=0.1$')
pylab.plot(feature_lengths,acq_r_2, '-o', color='r', label='$\lambda=0.5$')
pylab.plot(feature_lengths,acq_r_3, '-o', color='g', label='$\lambda=0.9$')
pylab.legend(loc='lower left')
pylab.xlabel('Length of features')
pylab.ylabel('Precision')
pylab.title('Comparing recall with varying $\lambda$ values')
pylab.xticks(feature_lengths)
pylab.grid()
pylab.tight_layout()
pylab.show()