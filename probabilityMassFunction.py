from scipy.special import comb 
import math
def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * error ** k * (1-error) ** (n_classifier -k) for k in range(k_start, n_classifier+ 1)]
    return sum(probs)
ensemble_error(n_classifier=11, error=0.25)
print(ensemble_error(n_classifier=11, error=0.25))

import numpy as np
import matplotlib.pyplot as plt
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]
plt.plot(error_range, ens_errors, linewidth=2)
plt.plot(error_range, error_range, linestyle='--', label='Base error', linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()

import numpy as np
np.argmax(np.bincount([0, 0, 1], weights = [0.2, 0.2, 0.6]))

ex = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])
p = np.average(ex, axis=0, weights = [0.2, 0.2, 0.6])
print(p)
np.argmax(p)
print(np.argmax(p))

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator
class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def _init_(self, classifiers, vote = 'classlabel', weights=None):
        self.classifiers = classifiers 
        self.named_classfiers = {
            key: value for key,
            value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        if self.vote not in ('probability', 'classlabe'):
            raise ValueError(f"vote must be 'probability' "
                             f"or 'classlabel' "
                             f"; got (vote = {self.vote})")
        
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Number of classfiers and '
                             f' weights must be equal '
                             f'; bot {len(self.weights)} weights, {len(self.classifiers)} classifiers')
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self