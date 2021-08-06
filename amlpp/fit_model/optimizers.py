from typing import List, Callable

import pandas as pd
import tqdm

##############################################################################
class Optimizer(object):
    def __init__(self, X_train:pd.DataFrame, Y_train:pd.DataFrame, 
                       X_test:pd.DataFrame, Y_test:pd.DataFrame,
                       rating_func:Callable, tqdm_bar:tqdm.tqdm,
                       model:Callable, model_params:dict,
                       ):

        self.X_train, self.Y_train = X_train, Y_train
        self.X_test, self.Y_test = X_test, Y_test
        self.rating_func = rating_func
        self.tqdm_bar = tqdm_bar
        self.model = model
        self.model_params = model_params

class SklearnOptimizer(Optimizer):
    def __init__(self, X_train:pd.DataFrame, Y_train:pd.DataFrame, 
                       X_test:pd.DataFrame, Y_test:pd.DataFrame,
                       rating_func:Callable, tqdm_bar:tqdm.tqdm, 
                       model:dict, model_params:Callable):

        super().__init__(X_train, Y_train, X_test, Y_test, 
                         rating_func, tqdm_bar, model, model_params)

    def __call__(self, trial):
        model = self.model(**self.model_params(trial)).fit(self.X_train, self.Y_train)
        score = self.rating_func(self.Y_test, model.predict(self.X_test))
        self.tqdm_bar.update(score, model.__class__.__name__)
        return score

class LGBMOptimizer(Optimizer):
    def __init__(self, X_train:pd.DataFrame, Y_train:pd.DataFrame, 
                       X_test:pd.DataFrame, Y_test:pd.DataFrame,
                       rating_func:Callable, tqdm_bar:tqdm.tqdm, 
                       model:dict, model_params:Callable, columns_params:List[str] = {}):

        super().__init__(X_train, Y_train, X_test, Y_test, 
                         rating_func, tqdm_bar, model, model_params)
        self.columns_params = columns_params

    def __call__(self, trial):
        model = self.model(**self.model_params(trial))
        model.fit(self.X_train, self.Y_train, eval_set = [(self.X_test, self.Y_test)], **self.columns_params)
        score = self.rating_func(self.Y_test, model.predict(self.X_test)) 
        self.tqdm_bar.update(score, model.__class__.__name__)
        return score