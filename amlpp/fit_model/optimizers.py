from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from typing import List, Callable, Tuple

import pandas as pd
import numpy as np
import tqdm

##############################################################################
class Optimizer(object):
    def __init__(self, X:pd.DataFrame, Y:pd.DataFrame, 
                       model:Callable, model_params:dict,
                       add_params:dict = {}, tqdm_bar:tqdm.tqdm = None,
                       rating_func:str = 'r2',
                       cross_validation:bool = True,
                       k_fold:int = 5, greater_is_better = True,
                       test_size:float = 0.2):
        
        self.model_params = model_params
        self.add_params = add_params
        self.tqdm_bar = tqdm_bar
        self.model = model
        self.X = X
        self.Y = Y

        self.greater_is_better = greater_is_better
        self.cross_validation = cross_validation
        self.rating_func = rating_func
        self.test_size = test_size
        self.k_fold = k_fold
    
    def __call__(self, trial):
        model = self.model(**self.model_params(trial))
        if type(self.rating_func) == str:
            scoring = make_scorer(get_scorer(self.rating_func)._score_func, greater_is_better = self.greater_is_better)
        else:
            scoring = make_scorer(self.rating_func, greater_is_better = self.greater_is_better)

        if self.cross_validation:
            score = cross_val_score(model, self.X, self.Y, cv = self.k_fold, scoring = scoring, fit_params = self.add_params)
            score = np.mean([sc for sc in score if sc == sc])
            if score != score:
                raise ValueError('Too small sample for cross validation')
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = self.test_size, random_state = 42)
            score = scoring(model.fit(X_train, Y_train), X_test, Y_test)

        if tqdm != None:
            self.tqdm_bar.update(score, model.__class__.__name__)
        return score