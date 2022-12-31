import pytest

from time import time
from sklearn_deap import EvolutionaryAlgorithmSearchCV, maximize
import sklearn.datasets
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import random


def func(x, y, m=1.0, z=False):
    return m * (np.exp(-(x ** 2 + y ** 2)) + float(z))


def readme():
    data = sklearn.datasets.load_digits()
    X = data['data']
    y = data['target']

    paramgrid = {
        'kernel': ['rbf'],
        'C': np.logspace(-9, 9, num=25, base=10),
        'gamma': np.logspace(-9, 9, num=25, base=10),
    }

    random.seed(1)

    cv = EvolutionaryAlgorithmSearchCV(
        estimator=SVC(),
        params=paramgrid,
        scoring='accuracy',
        cv=StratifiedKFold(n_splits=4),
        verbose=1,
        population_size=10,
        gene_mutation_prob=0.10,
        gene_crossover_prob=0.5,
        tournament_size=3,
        generations_number=5,
    )
    cv.fit(X, y)
    return cv


def test_cv():
    time1 = time()
    cv = readme()
    time2 = time() - time1

    print(f'Optimization run in {time2}s')
    cv_results_ = cv.cv_results_
    print('CV Results:\n{cv_results_}')
    assert cv_results_ is not None, 'cv_results is None.'
    assert cv_results_ != {}, 'cv_results is empty.'
    assert (
        cv.best_score_ == pytest.approx(1.0, 0.05),
        f'Did not find the best score. Returned: {cv.best_score_}'
    )


def test_optimize():
    ''' Simple hill climbing optimization with some twists. '''

    param_grid = {'x': [-1.0, 0.0, 1.0], 'y': [-1.0, 0.0, 1.0], 'z': [True, False]}
    args = {'m': 1.0}

    best_params, best_score, score_results, _, _ = maximize(
        func, param_grid, args, verbose=True
    )
    print('Score Results:\n{score_results}')
    assert best_params == {'x': 0.0, 'y': 0.0, 'z': True}
    assert best_score == 2.0
