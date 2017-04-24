import pandas as pd
import numpy as np
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from scipy.stats import mode


def smoothening(p):
	L = len(p)
	output = []
	step = 7
	for i in range(L):
		if (i < step):
			output.append(p[i])
		else:
			m = mode(p[range(i-step, i)])[0][0]
			output.append(m)
	return output
