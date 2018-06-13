import numpy as np
import copy
from sklearn.base import BaseEstimator, RegressorMixin

class Lasso(BaseEstimator, RegressorMixin):
	def __init__(self, alpha, max_iter):
		self.alpha = alpha				# 正則化項
		self.max_iter = max_iter			# 繰り返し回数
		self.coef_ = None				# 回帰係数
		self.intercept_ = None				# 切片係数

	def _soft_thresholding_operator(self, x, lambda_):
		if x > lambda_:
			return x - lambda_
		elif x < lambda_:
			return x + lambda_
		else:
			return 0

	beta = np.zeros(X.shape[1])

	for iteration in range(self.max_iter):
		for j in range(len(beta)):
			beta_tmp = copy.deepcopy(beta)
			beta_tmp[j] = 0
			r_j = y - np.dot(X, beta_tmp)
			arg1 = np.dot(X[:, j], r_j)
			arg2 = self.alpha*X.shape[0]
		beta[j] = self._soft_thresholding_operator(arg1, arg2)/(X[:, j]**2).sum()

	self.coef_ = beta

	def predict(self, X):
		y = np.dot(X, self.coef_)
		if self.fit_intercept:
			y += self.intercept_*np.ones(len(y))
		return y
