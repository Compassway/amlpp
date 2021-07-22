from sklearn.metrics import r2_score

from lightgbm import LGBMRegressor

from typing import List, Callable

import pandas as pd
import os

class LGBOptimizer(object):
    def __init__(self, X_train:pd.DataFrame, Y_train:pd.DataFrame, 
                       X_test:pd.DataFrame, Y_test:pd.DataFrame,
                       params_columns:List[str],
                       rating_func:Callable,
                       quantity_trials:int
                       ):

        self.X_train, self.Y_train = X_train, Y_train
        self.X_test, self.Y_test = X_test, Y_test
        self.params_columns = params_columns
        self.rating_func = rating_func

        self.iter = 0
        self.quantity_trials = quantity_trials

    def __call__(self, trial):
        params = {
        'metric': 'rmse', 
        'random_state':42,
        'n_estimators': 10000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.75,0.8,0.85]),
        'subsample': trial.suggest_categorical('subsample', [0.6,0.65,0.7,0.75,0.8,0.85]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.005,0.006,0.008,0.01,0.015,0.02,0.03]),
        'max_depth': trial.suggest_categorical('max_depth', [-1,10,20]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('cat_smooth', 1, 100)
        }
        
        model = LGBMRegressor(**params)

        model.fit(self.X_train, self.Y_train, eval_set = [(self.X_test, self.Y_test)], verbose = False, 
                **self.params_columns, early_stopping_rounds = 300)

        self.iter += 1
        progress = round(self.iter/self.quantity_trials*100, 2)
        score = self.rating_func(self.Y_test, model.predict(self.X_test))
        print(f"[{self.iter}/{self.quantity_trials} - {progress}%] {self.rating_func.__name__} = {round(score,3)}, trials № {trial.number}")
        return self.rating_func(self.Y_test, model.predict(self.X_test))