from lightgbm import LGBMRegressor

from typing import List, Callable

import pandas as pd
import os
import tqdm

class Optimizer(object):
    def __init__(self, X_train:pd.DataFrame, Y_train:pd.DataFrame, 
                       X_test:pd.DataFrame, Y_test:pd.DataFrame,
                       rating_func:Callable,
                       
                       params:List[str] = {}
                       ):

        self.X_train, self.Y_train = X_train, Y_train
        self.X_test, self.Y_test = X_test, Y_test
        self.rating_func = rating_func
        self.add_params = params
        self.tqdm_bar = tqdm_bar

class SklearnOptimizer(Optimizer):
    def __init__(self, X_train:pd.DataFrame, Y_train:pd.DataFrame, 
                       X_test:pd.DataFrame, Y_test:pd.DataFrame,
                       rating_func:Callable, quantity_trials:int,
                       tqdm_bar:object, model:dict, model_params:Callable, params:List[str] = {}, ):

        super().__init__(X_train, Y_train, X_test, Y_test, 
                        rating_func, quantity_trials, params)
        self.model = model
        self.model_params = model_params
        

    def __call__(self, trial):
        model = self.model(**self.model_params(trial)).fit(self.X_train, self.Y_train)
        self.tqdm.up()
        return self.rating_func(self.Y_test, model.predict(self.X_test))

class LGBMOptimizer(Optimizer):
    def __init__(self, X_train:pd.DataFrame, Y_train:pd.DataFrame, 
                       X_test:pd.DataFrame, Y_test:pd.DataFrame,
                       rating_func:Callable, quantity_trials:int,
                       model:dict, model_params:Callable, params:List[str] = {}):

        super().__init__(X_train, Y_train, X_test, Y_test, 
                        rating_func, quantity_trials, params)
        self.model = model
        self.model_params = model_params

    def __call__(self, trial):
        model = LGBMRegressor(**self.model_params(trial))
        model.fit(self.X_train, self.Y_train, eval_set = [(self.X_test, self.Y_test)], verbose = False, 
                **self.add_params, early_stopping_rounds = 300)
        self.pb.up()
        return self.rating_func(self.Y_test, model.predict(self.X_test))